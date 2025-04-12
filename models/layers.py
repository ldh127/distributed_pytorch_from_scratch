import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from models.comm_ops import Split, Reduce
from models.comm_ops import Copy, Gather
import process_manager as pm


class RowParallelLinear(nn.Module):
    def __init__(
            self, idim: int, odim: int, add_bias: bool = True,
            split_input: bool = True
    ):
        # forward: (b, idim) -[split]> (b, idim/n) -[linear]> (b, odim) -[gather]> (b, odim)
        # weight shape: (idim/n, odim)
        super().__init__()
        self.idim, self.odim = idim, odim
        self.split_input = split_input
        assert idim % pm.pgm.tp_size == 0
        self.idim_partition = self.idim // pm.pgm.tp_size
        self.weight = nn.Parameter(torch.Tensor(self.odim, self.idim_partition))
        self.add_bias = add_bias
        if add_bias:
            self.bias = nn.Parameter(torch.Tensor(self.odim))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def reset_parameters(self):
        weight = torch.empty(self.odim, self.idim, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))    # the same as pytorch default
        if pm.pgm.tp_size > 1:
            dist.broadcast(weight, src=0)       # broadcast weight to all processes to ensure weight consistency
            weight = torch.split(weight, self.idim_partition, dim=-1)[pm.pgm.tp_rank]   # (odim, idim) -> [(odim, idim/n), ...]
        self.weight.copy_(weight.contiguous())
        if self.add_bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 (optional): split the input tensor along the last dim, i.e. (..., idim) -> (..., idim/n)
        if self.split_input:
            x = Split.apply(x)
        # Step 2: linear
        x = F.linear(x, self.weight)
        # Step 3: reduce linear outputs, i.e. \sum_i^n(..., odim) -> (..., odim)
        x = Reduce.apply(x)
        # Step 4 (optional): add bias
        if self.add_bias:
            x = x + self.bias
        return x


class ColumnParallelLinear(nn.Module):
    def __init__(
            self, idim: int, odim: int, add_bias: bool = True,
            gather_output: bool = True
    ):
        # forward: (b, idim) -[linear]> (b, odim/n) -[gather]> (b, odim)
        # weight shape: (idim/n, odim)
        super().__init__()
        self.idim, self.odim = idim, odim
        self.gather_output = gather_output

        assert odim % pm.pgm.tp_size == 0
        self.odim_partition = self.odim // pm.pgm.tp_size
        self.weight = nn.Parameter(torch.Tensor(self.odim_partition, self.idim))
        self.add_bias = add_bias
        if add_bias:
            self.bias = nn.Parameter(torch.Tensor(self.odim_partition))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def reset_parameters(self):
        weight = torch.empty(self.odim, self.idim, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))    # the same as pytorch default
        if pm.pgm.tp_size > 1:
            dist.broadcast(weight, src=0)       # broadcast weight to all processes to ensure weight consistency
            weight = torch.split(weight, self.odim_partition, dim=0)[pm.pgm.tp_rank]   # (odim, idim) -> [(odim/n, idim), ...]
        self.weight.copy_(weight.contiguous())
        if self.add_bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: copy (this is for backward correctness)
        x = Copy.apply(x)
        # Step 2: linear, (..., idim) -> (..., odim/n)
        x = F.linear(x, self.weight)
        # Step 3 (optional): add bias
        if self.add_bias:
            x = x + self.bias
        # Step 4: gather linear outputs, i.e. [(.., odim/n), ...] -> (..., odim)
        if self.gather_output:
            x = Gather.apply(x)
        return x


class ParallelVocabularyEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hdim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hdim = hdim
        self.vocab_st_idx, self.vocab_ed_idx = self._get_vocab_range(vocab_size)
        self.weight = nn.Parameter(torch.Tensor(self.vocab_ed_idx - self.vocab_st_idx, self.hdim))

    @torch.no_grad()
    def reset_parameters(self):
        weight = torch.empty(self.vocab_size, self.hdim, device=self.weight.device, dtype=self.weight.dtype, requires_grad=False)
        nn.init.normal_(weight, mean=0., std=1.)    # the same as pytorch default
        if pm.pgm.tp_size > 1:
            dist.broadcast(weight, src=0)
            weight = torch.split(weight, self.vocab_size // pm.pgm.tp_size)[pm.pgm.tp_rank]
        self.weight.copy_(weight.contiguous())

    def _get_vocab_range(self, vocab_size: int) -> Tuple[int, int]:
        tp_size, tp_rank = pm.pgm.tp_size, pm.pgm.tp_rank
        assert tp_size < vocab_size
        n_vocab_per_partition = vocab_size // tp_size
        st_idx = n_vocab_per_partition * tp_rank
        ed_idx = st_idx + n_vocab_per_partition
        if pm.pgm.tp_rank == pm.pgm.tp_size - 1 and ed_idx != vocab_size:
            print(
                f"[WARNING]: In ParallelVocabularyEmbedding, the vocab size {vocab_size} is not divisible by tp_size {tp_size}. "
                f"The last partition will have {vocab_size - st_idx} embeddings."
            )
            ed_idx = vocab_size
        return st_idx, ed_idx

    def forward(self, x: torch.Tensor):
        # x: (B, L)
        assert x.ndim == 2, f"Input should be 2D tensor (B, L), but got {x.ndim}D tensor."
        m = torch.logical_and(self.vocab_st_idx <= x, x < self.vocab_ed_idx)
        x[m], x[~m] = x[m] - self.vocab_st_idx, 0
        out = F.embedding(x, self.weight)
        out[~m] = 0.                # will be gathered from other TP devices
        return Reduce.apply(out)    # gather from all TP devices


# Borrowed from LLama: https://github.com/meta-llama/llama/blob/main/llama/model.py#L34C1-L77
class RMSNorm(nn.Module):
    def __init__(self, hdim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hdim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self._norm(x.float()).type_as(x)
