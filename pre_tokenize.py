import os
import json
from argparse import ArgumentParser, Namespace

import tqdm

from tokenizers import Tokenizer
from constants import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, required=True)
    parser.add_argument('--output_file', '-o', type=str, required=True)
    parser.add_argument('--tokenizer_file', '-t', type=str, required=True)
    parser.add_argument('--splits', '-s', type=str, nargs='+', default=['train', 'validation'])
    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.exists(args.input_file), f'Input file {args.input_file} not found'
    with open(args.input_file, 'r') as f:
        datas = json.load(f)
    assert all(split in datas for split in args.splits), f'Expected splits {args.splits}, but found {list(datas.keys())}'

    assert os.path.exists(args.tokenizer_file), f'Tokenizer file {args.tokenizer_file} not found'
    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    token_data = {}
    for split in args.splits:
        token_data[split] = []
        token_lens = []
        for text in tqdm.tqdm(datas[split], desc=f'Tokenizing {split}'):
            tokens = tokenizer.encode(text).ids
            token_data[split].append(tokens)
            token_lens.append(len(tokens))
        print(
            f"Split: {split} -> "
            f"Number of samples: {len(token_data[split])}. "
            f"Max num_tokens: {max(token_lens)}. "
            f"Avg num_tokens: {sum(token_lens) / len(token_lens):.2f}."
        )
    token_data["special_ids"] = {
        BOS_TOKEN: tokenizer.token_to_id(BOS_TOKEN),
        EOS_TOKEN: tokenizer.token_to_id(EOS_TOKEN),
        UNK_TOKEN: tokenizer.token_to_id(UNK_TOKEN),
    }
    token_data["vocab_size"] = tokenizer.get_vocab_size()

    os.makedirs(os.path.dirname(args.output_file) or "./", exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(token_data, f, ensure_ascii=False)


if __name__ == '__main__':
    main()