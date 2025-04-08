import os
import math
import unittest
import argparse
from argparse import ArgumentParser
from typing import Tuple

import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from models.layers import ParallelVocabularyEmbedding, ColumnParallelLinear
import process_manager as pm


class ParallelToyModel(nn.Module):
    def __init__(self, vocab_size: int, idim: int, odim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.idim, self.odim = idim, odim
        self.vocab_embed = ParallelVocabularyEmbedding(vocab_size, idim)
        self.linear = ColumnParallelLinear(idim, odim, add_bias=True)
    
    def reset_parameters(self):
        self.linear.reset_parameters()
        self.vocab_embed.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"Expected 2D input (batch_size, seq_len), but got {x.shape}"
        x = self.vocab_embed(x)
        x = self.linear(x)
        return x        # (batch_size, seq_len, odim)


class VallinaToyModel(nn.Module):
    def __init__(self, vocab_size: int, idim: int, odim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.idim, odim = idim, odim
        self.vocab_embed = nn.Embedding(vocab_size, idim)
        self.linear = nn.Linear(idim, odim, bias=True)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear.bias)
        nn.init.normal_(self.vocab_embed.weight, mean=0., std=1.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"Expected 2D input (batch_size, seq_len), but got {x.shape}"
        x = self.vocab_embed(x)
        x = self.linear(x)
        return x        # (batch_size, seq_len, odim)


class TestParallelVocabularyEmbedding(unittest.TestCase):
    def __init__(self, random_seed: int, tp_size: int, rank: int):
        super().__init__()
        self.tp_size = tp_size
        self.random_seed = random_seed
        self.rank = rank
        # set random seed for reproducibility
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        self.init_random_state = torch.get_rng_state()      # save the initial random state for reuse
        self.init_cuda_random_state = torch.cuda.get_rng_state()

    def initialize_random_states(self):
        # init random states to the initial state so that parallel and non-parallel linear have the same random weights
        torch.set_rng_state(self.init_random_state)
        torch.cuda.set_rng_state(self.init_cuda_random_state)

    def runTest(self):
        self.test_one_pass()
        self.test_multiple_passes()

    def test_one_pass(self):
        fn_args = []
        for vocab_size in (4, 64, 1280, 16384, 65536):
            for hdim in (2, 64, 512, 1024, 4096):
                fn_args.append((vocab_size, hdim))
        
        for v, h in tqdm.tqdm(fn_args, desc='test_one_pass'):
            parallel_vocab_embed = ParallelVocabularyEmbedding(v, h).cuda()
            self.initialize_random_states()
            parallel_vocab_embed.reset_parameters()
            vallina_vocab_embed = nn.Embedding(v, h).cuda()
            self.initialize_random_states()
            nn.init.normal_(vallina_vocab_embed.weight, mean=0., std=1.)
            # check 1: parallel and vallina have the same random weights
            st_pos = parallel_vocab_embed.vocab_st_idx
            ed_pos = parallel_vocab_embed.vocab_ed_idx
            self.assertTrue(torch.allclose(vallina_vocab_embed.weight[st_pos: ed_pos], parallel_vocab_embed.weight, atol=1e-6))

            for bs in (1, 8, 16, 32, 64):
                for seq_len in (1, 8, 16, 32, 64):
                    x = torch.randint(v, (bs, seq_len)).cuda()
                    x_clone = x.clone()     # parallel_vocab_embed will modify x, so we clone it for later use in vallina_linear
                    parallel_out = parallel_vocab_embed(x)
                    vallina_out = vallina_vocab_embed(x_clone)
                    # check 2: parallel and vallina have the same output
                    self.assertTrue(torch.allclose(parallel_out, vallina_out, atol=1e-6))
    
    def _compare_model_weights(self, para_model: ParallelToyModel, vallina_model: VallinaToyModel):
        vocab_st_pos = para_model.vocab_embed.vocab_st_idx
        vocab_ed_pos = para_model.vocab_embed.vocab_ed_idx
        self.assertTrue(torch.allclose(para_model.vocab_embed.weight, vallina_model.vocab_embed.weight[vocab_st_pos: vocab_ed_pos], atol=1e-4))
        op = para_model.linear.odim_partition
        lin_st_pos, lin_ed_pos = op * pm.pgm.tp_rank, op * (pm.pgm.tp_rank + 1)
        self.assertTrue(torch.allclose(para_model.linear.weight.data, vallina_model.linear.weight.data[lin_st_pos: lin_ed_pos], atol=1e-4))
        self.assertTrue(torch.allclose(para_model.linear.bias.data, vallina_model.linear.bias.data[lin_st_pos: lin_ed_pos], atol=1e-4))

    def test_multiple_passes(self):
        idim, odim = 512, 2048
        n_train_steps = 1000
        vocab = 16384

        parallel_toy_model = ParallelToyModel(vocab, idim, odim).cuda()
        self.initialize_random_states()
        parallel_toy_model.reset_parameters()
        vallina_toy_model = VallinaToyModel(vocab, idim, odim).cuda()
        self.initialize_random_states()
        vallina_toy_model.reset_parameters()

        # check 1: parallel and vallina have the same random weights
        self._compare_model_weights(parallel_toy_model, vallina_toy_model)
        # train both models
        parallel_toy_model, parallel_loss_history = self.train(parallel_toy_model, vocab, n_train_steps, 'parallel')
        vallina_toy_model, vallina_loss_history = self.train(vallina_toy_model, vocab, n_train_steps, 'vallina')
        # check 2: parallel and vallina have the same loss history
        self.assertTrue(torch.allclose(parallel_loss_history, vallina_loss_history, atol=1e-6))
        # check 3: parallel and vallina have the same weights after training
        self._compare_model_weights(parallel_toy_model, vallina_toy_model)

    def train(self, model: nn.Module, vocab_size: int, n_steps: int, tag: str) -> Tuple[nn.Module, torch.Tensor]:
        loss_history = []
        self.initialize_random_states()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for i in tqdm.tqdm(range(n_steps), desc=f"Training {tag}"):
            bs = torch.randint(1, 16, (1,)).item()
            seqlen = torch.randint(64, 256, (1,)).item()
            x = torch.randint(0, vocab_size, (bs, seqlen), requires_grad=False, device='cuda')
            y = model(x)
            loss = y.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        return model, torch.tensor(loss_history, device='cuda', dtype=torch.float32)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='25555')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--tp_size', type=int, default=2)
    return parser.parse_args()


def run_test(rank: int, args: argparse.Namespace):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.tp_size, rank=rank)
    pm.init_pgm(args.tp_size)

    test_case = TestParallelVocabularyEmbedding(args.random_seed, args.tp_size, rank)
    suite = unittest.TestSuite()
    suite.addTest(test_case)
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    args = get_args()
    mp.spawn(run_test, args=(args,), nprocs=args.tp_size, join=True)
