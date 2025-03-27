import torch
import torch.distributed as dist

import process_manager as pm


class Split(torch.autograd.Function):
    # forward: split the input along the last dimension. (..., idim) -> (..., idim/n)
    # backward: cat the upstream gradient along the last dim. (..., idim/n) -> (..., idim)

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        if pm.pgm.tp_size == 1:
            return input
        dim = input.size(-1)
        assert dim % pm.pgm.tp_size == 0
        input_list = torch.split(input, dim // pm.pgm.tp_size, dim=-1)
        output = input_list[pm.pgm.tp_rank]
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if pm.pgm.tp_size == 1:
            return grad_output
        grad_output_list = [grad_output.new_zeros(grad_output.size()) for _ in range(pm.pgm.tp_size)]
        dist.all_gather(grad_output_list, grad_output, group=pm.pgm.tp_group)
        output = torch.cat(grad_output_list, dim=-1)
        return output


class Reduce(torch.autograd.Function):
    # forward: reduce the input along the last dimension. (..., idim) -> (..., idim)
    # backward: directly return the upstream gradient

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        if pm.pgm.tp_size == 1:
            return input
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=pm.pgm.tp_group)
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class Copy(torch.autograd.Function):
    # forward: directly return the input
    # backward: reduce the grad_output
    # Notes: Copy is the opposite operation of Reduce
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if pm.pgm.tp_size == 1:
            return grad_output
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pm.pgm.tp_group)
        return grad_output
    

class Gather(torch.autograd.Function):
    # forward: gather the input along the last dimension: (..., idim/n) -> (..., idim)
    # backward: split the upstream grad_output along the last dim: (..., idim) -> (..., idim/n)
    # Notes: Gather is the opposite operation of Split

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        if pm.pgm.tp_size == 1:
            return input
        input_list = [input.new_zeros(input.size()) for _ in range(pm.pgm.tp_size)]
        input_list[pm.pgm.tp_rank] = input
        dist.all_gather(input_list, input, group=pm.pgm.tp_group)
        return torch.cat(input_list, dim=-1).contiguous()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if pm.pgm.tp_size == 1:
            return grad_output
        assert grad_output.size(-1) % pm.pgm.tp_size == 0
        grad_arr = torch.split(grad_output, grad_output.size(-1) // pm.pgm.tp_size, dim=-1)
        return grad_arr[pm.pgm.tp_rank].contiguous()
