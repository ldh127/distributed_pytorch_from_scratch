import os
import re
import json
import random
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--validation_parition', type=float, default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    data_path = args.data_path
    output_path = args.output_path
    validation_parition = args.validation_parition
    assert os.path.exists(data_path)

    min_char = 15        # a heuristic rule to filter out short sentences
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()
    extracted = []
    for part in re.findall(r'\s+(?:\d+)\s*\n([\s\S]*?)\s+(?:\d+)\s*\n', data, re.DOTALL):
        for l in re.split(r'[\.?!]', part):
            l = l.replace('\n', '').strip()
            if len(l) > min_char:
                extracted.append(l)
    
    random.shuffle(extracted)    
    train_num = int(len(extracted) * (1 - validation_parition))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'train': extracted[:train_num], 
                'validation': extracted[train_num:]
            }, 
            f, indent=2, ensure_ascii=False
        )
    print(
        f"Training samples: {train_num}. " 
        f"Validation samples: {len(extracted) - train_num}. "
        f"Training chars: {sum(len(d) for d in extracted[:train_num])}. "
        f"Validation chars: {sum(len(d) for d in extracted[train_num:])}"
    )
    