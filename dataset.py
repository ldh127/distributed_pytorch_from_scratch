import os
import json
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast

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
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        self.tokenizer.bos_token = BOS_TOKEN
        self.tokenizer.eos_token = EOS_TOKEN
        self.tokenizer.unk_token = UNK_TOKEN
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id
        self.unk = self.tokenizer.unk_token_id
        self.vocab_size = len(self.tokenizer)
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.data[idx], return_tensors='pt')['input_ids'][0]
        if len(tokens) > self.maxlen:
            print(f"Warning: sequence {idx} is longer than maxlen {self.maxlen}. Truncating...")
        tokens = tokens[:self.maxlen]       # clip to maxlen
        return tokens     # (seq_len,)


def collate_fn(batch, bos: int, eos: int, ignore_idx: int):
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
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_partial, num_workers=4
    )
    return dataloader
