from dataclasses import dataclass

BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
IGNORE_INDEX = -1


@dataclass
class ModelArgumments:
    attn_dim: int = 512
    ffn_dim: int = 2048
    num_heads: int = 8
    rope_theta: float = 10000.
    num_layers: int = 12
    vocab_size: int = 1024
    maxlen: int = 256
