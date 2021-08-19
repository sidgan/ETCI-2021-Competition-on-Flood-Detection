"""
Referenced from:
https://discuss.pytorch.org/t/right-ways-to-serialize-and-load-ddp-model-checkpoints/122719/3
"""
import torch
import torch.distributed as dist


def global_meters_all_avg(rank, world_size, *meters):
    tensors = [
        torch.tensor(meter, device=rank, dtype=torch.float32) for meter in meters
    ]
    for tensor in tensors:
        # each item of `tensors` is all-reduced starting from index 0 (in-place)
        dist.all_reduce(tensor)

    return [(tensor / world_size).item() for tensor in tensors]


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
