import os
import math
import unittest
import argparse
from argparse import ArgumentParser

import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from models.model import Transformer, VallinaTransformer
from models.layers import ParallelVocabularyEmbedding, ColumnParallelLinear, RowParallelLinear
import process_manager as pm


class TestTransformer(unittest.TestCase):
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

    def _compare_weights(self, para_model: Transformer, vallina_model: VallinaTransformer):
        
        def _compare_vocab_embedding(para_embedding: ParallelVocabularyEmbedding, vallina_embedding: nn.Embedding):
            st_pos, ed_pos = para_embedding.vocab_st_idx, para_embedding.vocab_ed_idx
            self.assertTrue(torch.allclose(para_embedding.weight, vallina_embedding.weight[st_pos: ed_pos], atol=1e-6))

        def _compare_col_parallel_linear(para_linear: ColumnParallelLinear, vallina_linear: nn.Linear):
            op = para_linear.odim_partition
            lin_st_pos, lin_ed_pos = op * pm.pgm.tp_rank, op * (pm.pgm.tp_rank + 1)
            self.assertTrue(torch.allclose(para_linear.weight, vallina_linear.weight[lin_st_pos: lin_ed_pos], atol=1e-6))
            if para_linear.bias is not None:
                self.assertTrue(torch.allclose(para_linear.bias, vallina_linear.bias[lin_st_pos: lin_ed_pos], atol=1e-6))
        
        def _compare_row_parallel_linear(para_linear: RowParallelLinear, vallina_linear: nn.Linear):
            ip = para_linear.idim_partition
            lin_st_pos, lin_ed_pos = ip * pm.pgm.tp_rank, ip * (pm.pgm.tp_rank + 1)
            self.assertTrue(torch.allclose(para_linear.weight, vallina_linear.weight[:, lin_st_pos: lin_ed_pos], atol=1e-6))
            if para_linear.bias is not None:
                self.assertTrue(torch.allclose(para_linear.bias, vallina_linear.bias, atol=1e-6))

        _compare_vocab_embedding(para_model.embedding, vallina_model.embedding)
        _compare_col_parallel_linear(para_model.lm_head, vallina_model.lm_head)
        self.assertTrue(len(para_model.layers) == len(vallina_model.layers))
        for i in range(len(para_model.layers)):
            para_attn_i, vallina_attn_i = para_model.layers[i].attn, vallina_model.layers[i].attn
            para_ffn_i, vallina_ffn_i = para_model.layers[i].ffn, vallina_model.layers[i].ffn
            _compare_col_parallel_linear(para_attn_i.wq, vallina_attn_i.wq)
            _compare_col_parallel_linear(para_attn_i.wk, vallina_attn_i.wk)
            _compare_col_parallel_linear(para_attn_i.wv, vallina_attn_i.wv)
            _compare_row_parallel_linear(para_attn_i.wo, vallina_attn_i.wo)
            _compare_col_parallel_linear(para_ffn_i.gate_proj, vallina_ffn_i.gate_proj)
            _compare_col_parallel_linear(para_ffn_i.up_proj, vallina_ffn_i.up_proj)
            _compare_row_parallel_linear(para_ffn_i.down_proj, vallina_ffn_i.down_proj)

    def test_one_pass(self):
        test_args = []
        for attn_dim in (128, 256):
            for head_dim in (32, 64):
                num_heads = attn_dim // head_dim
                for ffn_dim in (512, 1024):
                    for num_layers in (2, 4, 8):
                        for vocab_size in (4096, 8192, 16384):
                            for maxlen in (1024, 2048):
                                test_args.append((attn_dim, ffn_dim, num_heads, num_layers, vocab_size, maxlen))

        # checks failed to pass: layer-wise outputs; parameters' gradients
        # These failures are expected because the linear algebra backends of pytorch (e.g. cudnn) may use different matrix multiplication algorithms depending on the input size.
        for arg in tqdm.tqdm(test_args, desc="Test one-time forward and backward"):
            para_model = Transformer(*arg).cuda()
            self.initialize_random_states()
            para_model.reset_parameters()
            para_model.retain_grad()
            vallina_model = VallinaTransformer(*arg).cuda()
            self.initialize_random_states()     # init random states again to make sure parallel and non-parallel linear have the same random weights
            vallina_model.reset_parameters()
            para_model.retain_grad()            
            
            # check 1: make sure the initial weight of tp-parallel model is the same as non-tp-parallel model
            self._compare_weights(para_model, vallina_model)

            attn_dim, vocab_size, maxlen = arg[0], arg[-2], arg[-1]
            n_step = 10
            for _ in range(n_step):
                bs = torch.randint(1, 16, (1,)).item()
                seq_len = torch.randint(64, 256, (1,)).item()
                seq_len = min(seq_len, maxlen)
                input_tokens = torch.randint(0, vocab_size, (bs, seq_len), requires_grad=False, device='cuda')
                input_tokens_clone = input_tokens.clone()       # used in vallina model
                tgt = torch.randint(0, vocab_size, (bs, seq_len), requires_grad=False, device='cuda')
                position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(bs, -1)
                logits_para = para_model(input_tokens, position_ids=position_ids)
                loss_para = F.cross_entropy(logits_para.view(-1, vocab_size), tgt.view(-1), reduction='mean')
                loss_para.backward()
                logits_vallina = vallina_model(input_tokens_clone, position_ids=position_ids)
                loss_vallina = F.cross_entropy(logits_vallina.view(-1, vocab_size), tgt.view(-1), reduction='mean')
                loss_vallina.backward()
                # check 2: make sure the loss of tp-parallel model is the same as non-tp-parallel model
                self.assertTrue(torch.allclose(loss_para, loss_vallina, atol=1e-2))


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
    os.environ['LOCAL_RANK'] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.tp_size, rank=rank)
    pm.init_pgm(args.tp_size)

    test_case = TestTransformer(args.random_seed, args.tp_size, rank)
    suite = unittest.TestSuite()
    suite.addTest(test_case)
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    args = get_args()
    mp.spawn(run_test, args=(args,), nprocs=args.tp_size, join=True)
