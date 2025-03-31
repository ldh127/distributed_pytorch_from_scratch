import os
import glob
import re
import random
import math
from dataclasses import dataclass, asdict
from argparse import ArgumentParser, Namespace

import tqdm
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from models.model import Transformer, VallinaTransformer
from dataset import get_dataloader
from constants import IGNORE_INDEX
import process_manager as pm


def get_train_args():
    parser = ArgumentParser()
    
    group = parser.add_argument_group("distributed")
    group.add_argument('--tp_size', type=int, default=2)
    group.add_argument('--master_addr', type=str, default='localhost')
    group.add_argument('--master_port', type=str, default='25555')
    
    group = parser.add_argument_group("training")
    group.add_argument("--lr", type=float, default=3e-4)
    group.add_argument("--warmup_steps", type=int, default=2000)
    group.add_argument("--max_steps", type=int, default=20000)
    group.add_argument("--log_interval", type=int, default=100)
    group.add_argument("--save_interval", type=int, default=1000)
    group.add_argument("--save_dir", type=str, default='./checkpoints')
    group.add_argument("--reserv_last_n_ckpts", type=int, default=5)
    group.add_argument("--batch_size", "-b", type=int, default=64)

    group = parser.add_argument_group("data")
    group.add_argument("--data_path", "-d", type=str, required=True)
    group.add_argument("--tokenizer_path", "-t", type=str, required=True)
    
    group = parser.add_argument_group("other")
    group.add_argument('--random_seed', type=int, default=0)
    group.add_argument('--use_vallina_impl', action='store_true', help="Whether to use vanilla implementation of transformer or not.")

    return parser.parse_args()


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


@dataclass
class ModelArgumments:
    attn_dim: int = 512
    ffn_dim: int = 2048
    num_heads: int = 8
    rope_theta: float = 10000.
    num_layers: int =  12
    vocab_size: int = 1024
    maxlen: int = 2048


def train(rank: int, args: Namespace):
    set_seed(args.random_seed)
    init_dist_env(args, rank)
    
    model_cls = VallinaTransformer if args.use_vallina_impl else Transformer
    model_args = ModelArgumments()
    model = model_cls(**asdict(model_args))
    model.cuda()
    model.reset_parameters()        # re-initialize parameters (neccassary for tensor-parallel Transformer)
    
    dataloader = get_dataloader(
        args.data_path, args.tokenizer_path, 
        args.batch_size, IGNORE_INDEX, split='train', maxlen=model_args.maxlen, shuffle=True,
    )
    assert dataloader.dataset.vocab_size == model_args.vocab_size, "vocab size of dataset and model should be the same"
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.max_steps, pct_start=args.warmup_steps / args.max_steps)
    summary_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, f"tprank-{rank}"))

    tag = f"TP-{pm.pgm.tp_rank}" if not args.use_vallina_impl else "vanilla"
    pbar = tqdm.tqdm(range(args.max_steps), desc=f"Training-[{tag}]", disable=rank != 0)
    accum_loss = 0.
    max_epoch = math.ceil(args.max_steps / len(dataloader))
    dist.barrier()
    for epoch in range(max_epoch):
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].cuda()
            target_ids = batch['target_ids'].cuda()
            position_ids = batch['position_ids'].cuda()
            logits = model(input_ids, position_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target_ids.view(-1), 
                ignore_index=IGNORE_INDEX, reduction='mean',
            )

            # debug
            if loss > 10 and pm.pgm.tp_rank == 0:
                print(f"[TP Rank: {pm.pgm.tp_rank}]: loss spike: {loss.item()}")
            
            dist.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            accum_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix({'avg_loss': accum_loss / pbar.n})
            if pbar.n % args.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                avg_loss = accum_loss / pbar.n
                print(f"[TP rank {rank}]: Step {pbar.n}/{args.max_steps} -> Avg Loss {avg_loss:.4f}, Lr {lr:.8f}")
                summary_writer.add_scalar('train/ce_loss', avg_loss, pbar.n)
                summary_writer.add_scalar('train/lr', lr, pbar.n)
            if pbar.n % args.save_interval == 0:
                avg_loss = accum_loss / pbar.n
                save_path = os.path.join(args.save_dir, f"tprank-{rank}_iter-{pbar.n}_loss-{avg_loss:.4f}.pth")
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"[TP rank {rank}]: Model saved to {save_path}")
                if args.reserv_last_n_ckpts > 0:
                    ckpts = glob.glob(os.path.join(args.save_dir, f"tprank-{rank}_iter-*_loss-*.pth"))
                    ckpts = sorted(ckpts, key=lambda x: int(re.findall(r'tprank-\d+_iter-(\d+)_loss-.+.pth', os.path.basename(x))[0]))
                    for ckpt in ckpts[:-args.reserv_last_n_ckpts]:
                        os.remove(ckpt)
                dist.barrier()
            if pbar.n >= args.max_steps:
                dist.barrier()
                print(f"[TP rank {rank}]: Training finished (total steps: {pbar.n}). Exiting...")
                break
    pbar.close()


if __name__ == '__main__':
    args = get_train_args()
    mp.spawn(train, args=(args, ), nprocs=args.tp_size, join=True)
