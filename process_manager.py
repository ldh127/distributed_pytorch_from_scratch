import torch
import torch.distributed as dist

global pgm
pgm = None


class ProcessGroupManager(object):
    def __init__(self, tp_size: int):
        assert dist.is_initialized()
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        assert tp_size == self.world_size
        self.tp_size = tp_size
        self.tp_rank = self.global_rank  # no other parallel is enabled (e.g. Data parallel), so tp_rank is equal to global_rank
        self.grid = torch.arange(self.world_size).view(self.tp_size)
        self.tp_group = dist.new_group(self.grid.tolist())

    def __repr__(self):
        return f"TP{self.tp_size}"


def init_pgm(tp_size: int):
    global pgm
    pgm = ProcessGroupManager(tp_size)
