import os
import json
from argparse import ArgumentParser

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel

from constants import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", "-d", type=str, required=True)
    parser.add_argument("--vocab_size", "-v", type=int, default=30000)
    parser.add_argument("--output_path", "-o", type=str, required=True)
    return parser.parse_args()


def get_json_iterator(data_path: str, split: str):
    with open(data_path, "r") as f:
        data = json.load(f)
        for d in data[split]:
            yield d


if __name__ == "__main__":
    args = get_args()

    data_iterator = get_json_iterator(args.data_path, "train")
    special_tokens = [BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = ByteLevel()
    
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens
    )
    tokenizer.train_from_iterator(data_iterator, trainer=trainer)

    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    unk_id = tokenizer.token_to_id(UNK_TOKEN)
    print(f"BOS token ID: {bos_id}")
    print(f"EOS token ID: {eos_id}")
    print(f"UNK token ID: {unk_id}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    tokenizer.save(args.output_path)
    print(f"Tokenizer saved to {args.output_path}")
