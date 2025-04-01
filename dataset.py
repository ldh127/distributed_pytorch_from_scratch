import os
import json
from functools import partial
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from constants import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN


class ShakespeareDataset(Dataset):
    def __init__(self, data_path: str, tokenizer_path: str, split: str, maxlen):
        assert split in ["train", "validation"], f"Expectected split to be 'train' or 'validation', but got {split}"
        assert os.path.exists(data_path) and os.path.exists(tokenizer_path)
        with open(data_path, 'r') as f:
            data = json.load(f)
            if split not in data:
                raise ValueError(f"Split {split} not found in data file {data_path}. Available splits: {data.keys()}")
            self.data = data[split]
        self.maxlen = maxlen
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        assert self.tokenizer.add_special_tokens([BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]) == 0, "Some special tokens are not added"
        self.bos = self.tokenizer.token_to_id(BOS_TOKEN)
        self.eos = self.tokenizer.token_to_id(EOS_TOKEN)
        self.unk = self.tokenizer.token_to_id(UNK_TOKEN)
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> List[int]:
        tokens = self.tokenizer.encode(self.data[idx]).ids
        if len(tokens) > self.maxlen:
            print(f"Warning: sequence {idx} is longer than maxlen {self.maxlen}. Truncating...")
            tokens = tokens[:self.maxlen]       # clip to maxlen
        return tokens     # (seq_len,)


def collate_fn(batch: List[List[int]], bos: int, eos: int, ignore_idx: int):
    batch = [torch.tensor(x) for x in batch]
    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len + 1), fill_value=eos, dtype=torch.long)
    target_ids = torch.full((len(batch), max_len + 1), fill_value=ignore_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, 0] = bos
        input_ids[i, 1: len(b) + 1] = b
        target_ids[i, :len(b)] = b
        target_ids[i, len(b)] = eos
    position_ids = torch.arange(max_len + 1, device=batch[0].device).unsqueeze(0).expand(len(batch), -1)
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'position_ids': position_ids,
    }


def get_dataloader(
    data_path: str, tokenizer_path: str, 
    batch_size: int, ignore_idx: int, split: str, maxlen: int, shuffle: bool = True
):
    dataset = ShakespeareDataset(data_path, tokenizer_path, split, maxlen=maxlen)
    collate_fn_partial = partial(collate_fn, bos=dataset.bos, eos=dataset.eos, ignore_idx=ignore_idx)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_partial, num_workers=0
    )
    return dataloader
