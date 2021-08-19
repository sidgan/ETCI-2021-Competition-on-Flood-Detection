import os
import torch
import numpy as np
import torch.distributed as dist


def init_process(rank, size, backend="gloo"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)


def seed_worker(id):
    # reset the random seed for every worker thread after an epoch
    seed = (id + torch.initial_seed()) % 2 ** 32
    np.random.seed(seed)
