import os
import random
from argparse import Namespace
import numpy as np

import torch
import torch.distributed as dist

import process_manager as pm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_dist_env(args: Namespace, rank: int):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.tp_size, rank=rank)
    pm.init_pgm(args.tp_size)
