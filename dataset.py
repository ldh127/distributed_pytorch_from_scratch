import os
import json
from functools import partial
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

from constants import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN


class ShakespeareDataset(Dataset):
    def __init__(self, data_path: str, split: str, maxlen):
        assert split in ["train", "validation"], f"Expectected split to be 'train' or 'validation', but got {split}"
        assert os.path.exists(data_path)
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            if split not in self.data:
                raise ValueError(f"Split {split} not found in data file {data_path}. Available splits: {self.data.keys()}")
        self.maxlen = maxlen
        self.split = split
        
        self.bos = self.data["special_ids"][BOS_TOKEN]
        self.eos = self.data["special_ids"][EOS_TOKEN]
        self.unk = self.data["special_ids"][UNK_TOKEN]
        self.vocab_size = self.data["vocab_size"]
        
    def __len__(self):
        return len(self.data[self.split])

    def __getitem__(self, idx) -> List[int]:
        tokens = self.data[self.split][idx]
        # clip to maxlen - 1. why -1: reserve one position for EOS/BOS
        if len(tokens) > self.maxlen - 1:
            print(f"Warning: sequence is longer than maxlen {self.maxlen - 1}: {len(tokens)}. Truncating...")
            tokens = tokens[:self.maxlen - 1]
        return tokens     # (seq_len,)


def collate_fn(batch: List[List[int]], bos: int, eos: int, ignore_idx: int):
    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len + 1), fill_value=eos, dtype=torch.long)
    target_ids = torch.full((len(batch), max_len + 1), fill_value=ignore_idx, dtype=torch.long)
    for i, b in enumerate(batch):
        input_ids[i, 0] = bos
        input_ids[i, 1: len(b) + 1] = torch.tensor(b, dtype=torch.long)
        target_ids[i, :len(b)] = torch.tensor(b, dtype=torch.long)
        target_ids[i, len(b)] = eos
    position_ids = torch.arange(max_len + 1, device=input_ids.device).unsqueeze(0).repeat(len(batch), 1)
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'position_ids': position_ids,
    }


def get_dataloader(
    data_path: str, 
    batch_size: int, ignore_idx: int, split: str, maxlen: int, shuffle: bool = True
):
    dataset = ShakespeareDataset(data_path, split, maxlen=maxlen)
    collate_fn_partial = partial(collate_fn, bos=dataset.bos, eos=dataset.eos, ignore_idx=ignore_idx)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_partial, 
        num_workers=0, pin_memory=True
    )
    return dataloader
