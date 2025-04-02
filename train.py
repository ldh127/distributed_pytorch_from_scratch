import os
import glob
import re
import math
from dataclasses import asdict
from argparse import ArgumentParser, Namespace

import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tensorboardX import SummaryWriter

from models.model import Transformer, VallinaTransformer
from dataset import get_dataloader
from constants import IGNORE_INDEX
from constants import ModelArgumments
import process_manager as pm
from utils import set_seed, init_dist_env


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
    group.add_argument("--bf16", action='store_true', help="Whether to use automatic mixed precision or not.")

    group = parser.add_argument_group("data")
    group.add_argument("--data_path", "-d", type=str, required=True)
    
    group = parser.add_argument_group("other")
    group.add_argument('--random_seed', type=int, default=0)
    group.add_argument('--use_vallina_impl', action='store_true', help="Whether to use vanilla implementation of transformer or not.")

    args = parser.parse_args()
    return args


def train(rank: int, args: Namespace):
    set_seed(args.random_seed)
    init_dist_env(args, rank)
    if args.bf16:
        print("Enable bf16 training")
        os.environ['DTYPE'] = 'bfloat16'
    else:
        print("Disable bf16 training")
        os.environ['DTYPE'] = 'float32'
    
    model_cls = VallinaTransformer if args.use_vallina_impl else Transformer
    model_args = ModelArgumments()
    model = model_cls(**asdict(model_args))
    model.cuda()
    model.reset_parameters()        # re-initialize parameters (neccassary for tensor-parallel Transformer)
    model.train()

    if pm.pgm.tp_rank == 0:
        print(model)
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_param / 1e6:.4f} million")
    
    dataloader = get_dataloader(
        args.data_path,
        args.batch_size, IGNORE_INDEX, split='train', maxlen=model_args.maxlen, shuffle=True,
    )
    assert dataloader.dataset.vocab_size == model_args.vocab_size, "vocab size of dataset and model should be the same"
    
    dist.barrier()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.max_steps, pct_start=args.warmup_steps / args.max_steps)
    scaler = GradScaler() if args.bf16 else None
    summary_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, f"tprank-{rank}"))

    tag = f"TP-{pm.pgm.tp_rank}" if not args.use_vallina_impl else "vanilla"
    pbar = tqdm.tqdm(range(args.max_steps), desc=f"Training-[{tag}]", position=rank)
    accum_loss = 0.
    max_epoch = math.ceil(args.max_steps / len(dataloader))
    dist.barrier()
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    for epoch in range(max_epoch):
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].cuda()
            target_ids = batch['target_ids'].cuda()
            position_ids = batch['position_ids'].cuda()
            with autocast(enabled=args.bf16, dtype=dtype):
                logits = model(input_ids, position_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1), 
                    ignore_index=IGNORE_INDEX, reduction='mean',
                )
            del batch, input_ids, target_ids, position_ids, logits
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
                break
        if pm.pgm.tp_rank == 0:
            print(f"Epoch {epoch + 1}/{max_epoch} finished.")
        if pbar.n >= args.max_steps:
            print(f"[TP rank {rank}]: Training finished (total steps: {pbar.n}). Exiting...")
            break
        torch.cuda.empty_cache()

    pbar.close()
    summary_writer.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    args = get_train_args()
    mp.spawn(train, args=(args, ), nprocs=args.tp_size, join=True)