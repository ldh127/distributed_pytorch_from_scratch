import os
import re
import glob
import argparse
from dataclasses import asdict

import tqdm
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

import process_manager as pm
from models.model import Transformer
from dataset import get_dataloader
from constants import IGNORE_INDEX, BOS_TOKEN, EOS_TOKEN
from constants import ModelArgumments
from utils import init_dist_env, set_seed


def get_test_args():
    parser = argparse.ArgumentParser()
    
    group = parser.add_argument_group("distributed")
    group.add_argument('--master_addr', type=str, default='localhost')
    group.add_argument('--master_port', type=str, default='23333')
    group.add_argument('--tp_size', type=int, default=2)

    group = parser.add_argument_group("data")
    group.add_argument("--data_path", "-d", type=str, required=True)
    group.add_argument("--tokenizer_path", "-t", type=str, required=True)
    
    group = parser.add_argument_group("model")
    group.add_argument("--use_vallina_impl", action='store_true')
    parser.add_argument("--ckpt_dir", type=str, required=True)
    
    group = parser.add_argument_group("decode")
    group.add_argument("--max_decode_len", type=int, default=128)

    group = parser.add_argument_group("other")
    group.add_argument("--random_seed", type=int, default=0)
    
    args = parser.parse_args()
    return args


def calc_loss(model: Transformer, ckpt_path: str, dataloader: DataLoader, dtype: torch.dtype) -> float:
    # load checkpoint
    model.eval()
    ckpt = torch.load(ckpt_path, map_location='cuda')
    for k in ckpt:
        ckpt[k] = ckpt[k].to(dtype)
    model.to(dtype)
    model.load_state_dict(ckpt)
    
    # calc validation loss
    pbar = tqdm.tqdm(range(len(dataloader)), desc=f"[TP Rank {pm.pgm.tp_rank}]: calc validation loss", position=pm.pgm.tp_rank)
    accum_loss = 0.
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        target_ids = batch['target_ids'].cuda()
        position_ids = batch['position_ids'].cuda()
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            logits = model(input_ids, position_ids)
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), target_ids.view(-1), 
            ignore_index=IGNORE_INDEX, reduction='mean',
        )
        accum_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix({'avg_loss': accum_loss / pbar.n})
    pbar.close()

    avg_loss = accum_loss / len(dataloader.dataset)
    return avg_loss


def test(rank, args):
    set_seed(args.random_seed)
    init_dist_env(args, rank)
    
    model_args = ModelArgumments()
    model = Transformer(**asdict(model_args))
    model.cuda()
    model.reset_parameters()        # re-initialize parameters (neccassary for tensor-parallel Transformer)
    model.eval()

    ckpt_paths = glob.glob(os.path.join(args.ckpt_dir, f'tprank-{pm.pgm.tp_rank}_iter-*_loss-*.pth'))
    ckpt_paths = sorted(ckpt_paths, key=lambda x: int(re.findall(r"tprank-\d+_iter-(\d+)_loss-.*?.pth", x)[0]))
    if len(ckpt_paths) == 0:
        raise ValueError(f"[TP Rank {pm.pgm.tp_rank}]: No checkpoints found in {args.ckpt_dir}")
    print(f"[TP Rank {pm.pgm.tp_rank}]: Found {len(ckpt_paths)} checkpoints. Loading the latest one: {ckpt_paths[-1]}")
    ckpt_path = ckpt_paths[-1]
    ckpt = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(ckpt)

    # use half precision for inference
    os.environ["DTYPE"] = "bfloat16"
    dtype = torch.bfloat16
    model.to(dtype)

    batch_size = 1
    dataloader = get_dataloader(
        args.data_path, batch_size, IGNORE_INDEX, 
        split='validation', maxlen=model_args.maxlen, shuffle=False,
    )

    # calc validation loss
    pbar = tqdm.tqdm(range(len(dataloader)), desc=f"[TP Rank {pm.pgm.tp_rank}]: calc validation loss", position=pm.pgm.tp_rank)
    accum_loss = 0.
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        target_ids = batch['target_ids'].cuda()
        position_ids = batch['position_ids'].cuda()
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            logits = model(input_ids, position_ids)
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), target_ids.view(-1), 
            ignore_index=IGNORE_INDEX, reduction='mean',
        )
        accum_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix({'avg_loss': accum_loss / pbar.n})

        # debug
        if pbar.n > 10:
            break

    pbar.close()

    avg_loss = accum_loss / len(dataloader.dataset)
    print(f"[TP Rank {pm.pgm.tp_rank}]: Validation loss: {avg_loss}")
    save_path = os.path.join(args.ckpt_dir, "val", f'tprank-{pm.pgm.tp_rank}_val.txt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if pm.pgm.tp_rank == 0:
        with open(save_path, 'a') as f:
            f.write(f"Ckpt -> Validation loss\n")
            f.write(f"{ckpt_path} -> {avg_loss}\n")

    # continue writing (greedy decoding)
    texts = [
        "Nice to meet you, it's",
        "Great empire never falls, it only",
        "Your majesty, it's my duty ",
        "I shall be glad ",
        "What a glory to ",
        "Shame for the weak, it's",
        "The brave man ne",
        "Poor old man, it's"
    ]
    decoded = []
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    bos_id = dataloader.dataset.bos
    eos_id = dataloader.dataset.eos
    assert tokenizer.token_to_id(BOS_TOKEN) == bos_id and tokenizer.token_to_id(EOS_TOKEN) == eos_id
    for t in texts:
        t = t.strip()
        tokens = torch.tensor(tokenizer.encode(t).ids, dtype=torch.long).cuda().view(1, -1)   # (1, seq_len)
        tokens = F.pad(tokens, (1, 0), mode="constant", value=bos_id)
        while True:
            position_ids = torch.arange(tokens.size(-1), dtype=torch.long, device=tokens.device).unsqueeze(0)    # (1, seq_len)
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                logits = model(tokens, position_ids)[0, -1]     # (vocab_size,)
            pred_token = logits.argmax(dim=-1).item()
            tokens = F.pad(tokens, (0, 1), mode="constant", value=pred_token)   # (1, seq_len + 1)
            if tokens[0, -1].item() == eos_id or tokens.size(-1) > args.max_decode_len:
                if tokens[0, -1] == eos_id:
                    tokens = tokens[:, :-1]   # remove EOS
                else:
                    print(f"[TP Rank {pm.pgm.tp_rank}]: Maximum length reached. Stop decoding.")
                trans = tokenizer.decode(tokens[0, 1:].tolist()).strip()     # [1:-1]: remove BOS and EOS
                assert t in trans, f"Prediction {trans} does not contain the input text {t}"
                decoded.append((t, trans[len(t):]))
                break
    
    if pm.pgm.tp_rank == 0:
        with open(save_path, 'a') as fp:
            print(f"Input texts -> Decoded texts", file=fp)
            for input_text, decoded_text in decoded:
                print(f"{input_text} -> {decoded_text}")
                print(f"{input_text} -> {decoded_text}", file=fp)


if __name__ == '__main__':
    args = get_test_args()
    mp.spawn(test, args=(args, ), nprocs=args.tp_size, join=True)
