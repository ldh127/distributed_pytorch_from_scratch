import os
import math
from typing import Tuple
from enum import Enum

import einops
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.layers import ParallelVocabularyEmbedding
from models.layers import RowParallelLinear, ColumnParallelLinear
from models.layers import RMSNorm
import process_manager as pm


# Copied from transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L151-L155
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


# Modified from transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L157-L161
def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert cos.ndim == 3 and sin.ndim == 3, f"Expected cos and sin to be 3D tensors (batch_size, seq_length, head_dim), got {cos.ndim} and {sin.ndim}"
    cos = einops.rearrange(cos, "b t d -> b 1 t d")
    sin = einops.rearrange(sin, "b t d -> b 1 t d")
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Mofified from picotron: https://github.com/huggingface/picotron/blob/main/picotron/model.py#L21-L31
def get_cos_sin(seq_length: int, head_dim: int, base: float) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0
    # Results on CUDA and CPU are different even with the same formula, To match transformers implementation. frequency should be computed on CPU
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to('cpu') / head_dim))
    dtype = torch.bfloat16 if os.getenv('DTYPE', 'bfloat16') == 'bfloat16' else torch.float32
    device = torch.device('cuda') if os.getenv('DEVICE', 'cuda') == 'cuda' else torch.device('cpu')
    position = torch.arange(seq_length).to(device).unsqueeze(1).float()     # [seq_length, 1]
    # To match transformers implementation. m * theta should be computed on GPU
    theta = theta.to(device)
    cos = torch.cos(position.float() * theta.float()).to(dtype).repeat(1, 2)
    sin = torch.sin(position.float() * theta.float()).to(dtype).repeat(1, 2)
    return cos, sin     # [seq_length, head_dim], [seq_length, head_dim]


class Attention(nn.Module):
    def __init__(self, attn_dim: int, num_heads: int):
        super().__init__()
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        assert num_heads % pm.pgm.tp_size == 0 and attn_dim % num_heads == 0
        self.num_local_heads = num_heads // pm.pgm.tp_size
        self.wq = ColumnParallelLinear(attn_dim, attn_dim, gather_output=False)
        self.wk = ColumnParallelLinear(attn_dim, attn_dim, gather_output=False)
        self.wv = ColumnParallelLinear(attn_dim, attn_dim, gather_output=False)
        self.wo = RowParallelLinear(attn_dim, attn_dim, split_input=False)

    def reset_parameters(self):
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.wo.reset_parameters()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.size()
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q, k, v = map(lambda x: einops.rearrange(x, "b t (n d) -> b n t d", n=self.num_local_heads), (q, k, v))
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(b, 1, t, t, device=q.device), diagonal=1).bool()   # causal mask
        attn_score.masked_fill_(mask, -10000.)
        attn = torch.softmax(attn_score, dim=-1)
        o = einops.rearrange(torch.matmul(attn, v), "b n t d -> b t (n d)")
        return self.wo(o)


class FFN(nn.Module):
    def __init__(self, idim: int, hdim: int):
        super().__init__()
        self.idim, self.hdim = idim, hdim
        self.gate_proj = ColumnParallelLinear(idim, hdim, gather_output=False)
        self.up_proj = ColumnParallelLinear(idim, hdim, gather_output=False)
        self.down_proj = RowParallelLinear(hdim, idim, split_input=False)

    def reset_parameters(self):
        self.gate_proj.reset_parameters()
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(
            self, attn_dim: int, ffn_dim: int, num_heads: int, 
            maxlen: int = 2048, rope_theta: float = 10000.
    ):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.head_dim = attn_dim // num_heads
        self.attn = Attention(attn_dim, num_heads)
        self.ffn = FFN(attn_dim, ffn_dim)
        self.norm1 = RMSNorm(attn_dim)
        self.norm2 = RMSNorm(attn_dim)
        self.cos, self.sin = get_cos_sin(maxlen, self.head_dim, rope_theta)     # (maxlen, head_dim)

    def reset_parameters(self):
        self.attn.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos[position_ids], self.sin[position_ids]       # (bs, seq_len, head_dim)   
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
            self, attn_dim: int, ffn_dim: int, num_heads: int, 
            num_layers: int, vocab_size: int, maxlen: int = 2048, rope_theta: float = 10000.,
    ):
        super().__init__()
        self.embedding = ParallelVocabularyEmbedding(vocab_size, attn_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(attn_dim, ffn_dim, num_heads, maxlen, rope_theta)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(attn_dim)
        self.lm_head = ColumnParallelLinear(attn_dim, vocab_size, gather_output=True)

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.lm_head.reset_parameters()

    def retain_grad(self):
        # This is only for the unit test to check the gradient flow.
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.retain_grad()

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, position_ids)
        logits = self.lm_head(self.norm(x))
        return logits


class VallinaTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replace_tp_module_with_vallina_module()
    
    def _replace_tp_module_with_vallina_module(self):
        vocab_size = self.embedding.vocab_size
        attn_dim = self.layers[0].attn.attn_dim
        ffn_dim = self.layers[0].ffn.hdim
        self.embedding = nn.Embedding(vocab_size, attn_dim)  
        self.lm_head = nn.Linear(attn_dim, vocab_size)
        for layer in self.layers:
            layer.attn.num_local_heads = layer.attn.num_heads
            layer.attn.wq = nn.Linear(attn_dim, attn_dim)
            layer.attn.wk = nn.Linear(attn_dim, attn_dim)
            layer.attn.wv = nn.Linear(attn_dim, attn_dim)
            layer.attn.wo = nn.Linear(attn_dim, attn_dim)
            layer.ffn.gate_proj = nn.Linear(attn_dim, ffn_dim)
            layer.ffn.up_proj = nn.Linear(attn_dim, ffn_dim)
            layer.ffn.down_proj = nn.Linear(ffn_dim, attn_dim)
    
    def reset_parameters(self):
        # To have exactly the same initialization as the tp implementation, the module init order should be the same as the tp implementation.
        self._init_vocab_embedding(self.embedding)
        for layer in self.layers:
            self._init_linear(layer.attn.wq)
            self._init_linear(layer.attn.wk)
            self._init_linear(layer.attn.wv)
            self._init_linear(layer.attn.wo)
            self._init_linear(layer.ffn.gate_proj)
            self._init_linear(layer.ffn.up_proj)
            self._init_linear(layer.ffn.down_proj)
        self._init_linear(self.lm_head)
    
    def _init_linear(self, module: nn.Linear):
        weight, bias = module.weight, module.bias
        bound = math.sqrt(2. / weight.size(1))
        nn.init.normal_(weight, -bound, bound)
        if bias is not None:
            nn.init.zeros_(bias)
    
    def _init_vocab_embedding(self, module: nn.Embedding):
        nn.init.normal_(module.weight, mean=0., std=1.)
