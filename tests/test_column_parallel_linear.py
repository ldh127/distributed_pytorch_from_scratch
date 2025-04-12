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

from models.layers import ColumnParallelLinear
import process_manager as pm


class TestColumnParallelLinear(unittest.TestCase):
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
        self.test_multiple_pass()

    def _compare_model_weights(self, col_parallel_linear: ColumnParallelLinear, vallina_linear: nn.Linear):
        odim_per_partition = col_parallel_linear.odim_partition
        st_pos = pm.pgm.tp_rank * odim_per_partition
        end_pos = (pm.pgm.tp_rank + 1) * odim_per_partition
        self.assertTrue(torch.allclose(vallina_linear.weight.data[st_pos: end_pos], col_parallel_linear.weight.data))
        if col_parallel_linear.add_bias:
            self.assertTrue(torch.allclose(vallina_linear.bias.data[st_pos: end_pos], col_parallel_linear.bias.data))

    def test_one_pass(self):
        test_args = []
        for idim in (512, 1024, 2048):
            for odim in (1024, 2048, 4096):
                for add_bias in (True, False):
                    test_args.append((idim, odim, add_bias))
        
        for idim, odim, add_bias in tqdm.tqdm(test_args, desc="Test one-time forward and backward"):
            col_parallel_linear = ColumnParallelLinear(idim, odim, add_bias=add_bias, gather_output=True).cuda()
            self.initialize_random_states()
            col_parallel_linear.reset_parameters()
            # why retain_grad: the weight of col_parallel_linear may not be a leaf node. To access its grad, we need to retain_grad.
            col_parallel_linear.weight.retain_grad()
            if add_bias:
                col_parallel_linear.bias.retain_grad()
            
            vallina_linear = nn.Linear(idim, odim, bias=add_bias).cuda()
            self.initialize_random_states()     # init random states again to make sure parallel and non-parallel linear have the same random weights
            nn.init.kaiming_uniform_(vallina_linear.weight, a=math.sqrt(5))
            vallina_linear.weight.retain_grad()
            if add_bias:
                nn.init.zeros_(vallina_linear.bias)
                vallina_linear.bias.retain_grad()
            
            # check 1: make sure the initial weight of col_parallel_linear is the same as vallina_linear
            self._compare_model_weights(col_parallel_linear, vallina_linear)

            for bs in (1, 8, 16):
                for seq_len in (32, 64, 128, 256):
                    x = torch.rand(bs, seq_len, idim, requires_grad=True, device='cuda')
                    x_clone = x.clone()
                    y_parallel = col_parallel_linear(x)
                    y_parallel.mean().backward()
                    x_parallel_grad = x.grad.clone()
                    x.grad.zero_()
                    w_parallel_grad = col_parallel_linear.weight.grad.clone()
                    col_parallel_linear.weight.grad.zero_()
                    if add_bias:
                        b_parallel_grad = col_parallel_linear.bias.grad.clone()
                        col_parallel_linear.bias.grad.zero_()
                    
                    y_vallina = vallina_linear(x_clone)
                    y_vallina.mean().backward()
                    x_vallina_grad = x.grad.clone()
                    x.grad.zero_()
                    w_vallina_grad = vallina_linear.weight.grad.clone()
                    vallina_linear.weight.grad.zero_()
                    if add_bias:
                        b_vallina_grad = vallina_linear.bias.grad.clone()
                        vallina_linear.bias.grad.zero_()

                    # check 2: make sure the forward results are the same
                    self.assertEqual(y_parallel.shape, y_vallina.shape)
                    # when atol is more strict than 1e-4, the following assertion may fail.
                    # This is likely due to the fact that linear may have different implementations of matrix multiplication for different sizes of input.
                    self.assertTrue(torch.allclose(y_parallel, y_vallina, atol=1e-4))
                    # check 3: make sure the backward results are the same
                    self.assertTrue(torch.allclose(x_parallel_grad, x_vallina_grad))
                    odim_per_partition = odim // pm.pgm.tp_size
                    st_pos = pm.pgm.tp_rank * odim_per_partition
                    end_pos = (pm.pgm.tp_rank + 1) * odim_per_partition
                    self.assertTrue(torch.allclose(w_parallel_grad, w_vallina_grad[st_pos: end_pos]))
                    if add_bias:
                        self.assertTrue(torch.allclose(b_parallel_grad, b_vallina_grad[st_pos: end_pos]))

    def test_multiple_pass(self):
        idim, odim = 512, 1024
        add_bias = True
        n_train_steps = 1000

        col_parallel_linear = ColumnParallelLinear(idim, odim, add_bias=add_bias, gather_output=True).cuda()
        self.initialize_random_states()
        col_parallel_linear.reset_parameters()

        vallina_linear = nn.Linear(idim, odim, bias=add_bias).cuda()
        self.initialize_random_states()
        nn.init.kaiming_uniform_(vallina_linear.weight, a=math.sqrt(5))
        if add_bias:
            nn.init.zeros_(vallina_linear.bias)
            
        # check 1: make sure the initial weight of col_parallel_linear is the same as vallina_linear
        self._compare_model_weights(col_parallel_linear, vallina_linear)

        # check 2: after multiple updates, the weight of parallel and non-parallel linear should be the same
        col_parallel_linear, para_loss_history = self.train(col_parallel_linear, idim, n_train_steps, "parallel")
        vallina_linear, vallina_loss_history = self.train(vallina_linear, idim, n_train_steps, "vallina")
        self._compare_model_weights(col_parallel_linear, vallina_linear)

        # check 3: the loss histories should be all the same for both parallel and non-parallel cases
        self.assertTrue(torch.allclose(vallina_loss_history, para_loss_history, atol=1e-6))

    def train(self, model: nn.Module, idim: int, n_steps: int, tag: str) -> nn.Module:
        loss_history = []
        self.initialize_random_states()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        for i in tqdm.tqdm(range(n_steps), desc=f"Training {tag}"):
            bs = torch.randint(1, 16, (1,)).item()
            seqlen = torch.randint(64, 256, (1,)).item()
            x = torch.rand(bs, seqlen, idim, requires_grad=True, device='cuda')
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

    test_case = TestColumnParallelLinear(args.random_seed, args.tp_size, rank)
    suite = unittest.TestSuite()
    suite.addTest(test_case)
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    args = get_args()
    mp.spawn(run_test, args=(args,), nprocs=args.tp_size, join=True)
