import math

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
            with torch.no_grad():       # init with zero
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        weight = torch.empty(self.odim, self.idim, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False)
        bound = math.sqrt(2. / weight.size(1))
        nn.init.normal_(weight, -bound, bound)
        if pm.pgm.tp_size > 1:
            dist.broadcast(weight, src=0)       # broadcast weight to all processes to ensure weight consistency
            weight_list = torch.split(weight, self.idim_partition, dim=-1)   # (odim, idim) -> [(odim, idim/n), ...]
            with torch.no_grad():
                self.weight.copy_(weight_list[pm.pgm.tp_rank].contiguous())
        else:
            with torch.no_grad():
                self.weight.copy_(weight.contiguous())

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
        # forward: (b, idim) -[split]> (b, idim/n) -[linear]> (b, odim) -[gather]> (b, odim)
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
            with torch.no_grad():       # init with zero
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        weight = torch.empty(self.odim, self.idim, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False)
        bound = math.sqrt(2. / weight.size(1))
        nn.init.normal_(weight, -bound, bound)
        if pm.pgm.tp_size > 1:
            dist.broadcast(weight, src=0)       # broadcast weight to all processes to ensure weight consistency
            weight_list = torch.split(weight, self.odim_partition, dim=0)   # (odim, idim) -> [(odim/n, idim), ...]
            with torch.no_grad():
                self.weight.copy_(weight_list[pm.pgm.tp_rank].contiguous())
        else:
            with torch.no_grad():
                self.weight.copy_(weight.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: copy (this is for backward correctness)
        x = Copy.apply(x)
        # Step 2: linear, (..., idim) -> (..., odim/n)
        x = F.linear(x, self.weight)
        # Step 3 (optional): add bias
        if self.add_bias:
            x = x + self.bias
        # Step 4: gather linear outputs, i.e. [(.., odim/n), ...] -> (..., odim)
        x = Gather.apply(x)
        return x
