import os
from typing import Any, Dict

import numpy as np
import torch
from torch import distributed as dist


def initialize_torch_distributed():
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    if torch.cuda.is_available():
        # Set the device id.
        # TODO @thomasw21: `torch.cuda.device_count` should return the number of device on a single node. We assume the nodes to be homogeneous (same number of gpus per node)
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        # TODO @thomasw21
        backend = "gloo"

    # Call the init process.
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        init_method=init_method
    )
    return True

def get_process_groups(
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> Dict[str, Any]:
    """
    Generate all the process groups necessary for training, and returning current ranks process groups.

    :param data_parallel_size: int
    :param tensor_parallel_size: int
    :param pipeline_parallel_size: int
    :return: DistributedProcessGroups
    """
    if not dist.is_available():
        raise ValueError("`torch.distributed is not available as a package, please install it.")

    if data_parallel_size * tensor_parallel_size * pipeline_parallel_size <= 1:
        raise ValueError("No need to use `brrr` if you're not using distributed training.")

    if not dist.is_initialized():
        initialize_torch_distributed()

    world_pg = dist.distributed_c10d._get_default_group()
    world_size = world_pg.size()
    world_rank = world_pg.rank()
    assert world_size == data_parallel_size * tensor_parallel_size * pipeline_parallel_size, \
        f"{world_size} != {data_parallel_size * tensor_parallel_size * pipeline_parallel_size}"

    # In the current implementation in DeepSpeed, tp then dp then pp
    #   https://cs.github.com/microsoft/DeepSpeed/blob/591744eba33f2ece04c15c73c02edaf384dca226/deepspeed/runtime/pipe/topology.py#L243

    ranks = np.arange(0, world_size).reshape((pipeline_parallel_size, data_parallel_size, tensor_parallel_size))

    tp_pg = None
    if tensor_parallel_size > 1:
        ranks_with_tp_last = ranks.reshape((pipeline_parallel_size * data_parallel_size, tensor_parallel_size))
        for tp_ranks in ranks_with_tp_last:
            new_group = torch.distributed.new_group(
                ranks=tp_ranks
            )
            # TODO @thomasw21: fix this as the `in` can be quite expensive
            if world_rank in tp_ranks:
                tp_pg = new_group

    dp_pg = None
    if data_parallel_size > 1:
        ranks_with_dp_last = ranks.transpose((0, 2, 1)).reshape((pipeline_parallel_size * tensor_parallel_size, data_parallel_size))
        for dp_ranks in ranks_with_dp_last:
            new_group = torch.distributed.new_group(
                ranks=dp_ranks
            )
            if world_rank in dp_ranks:
                dp_pg = new_group

    pp_pg = None
    if pipeline_parallel_size > 1:
        ranks_with_pp_last = ranks.transpose((1, 2, 0)).reshape((tensor_parallel_size * data_parallel_size, pipeline_parallel_size))
        for pp_ranks in ranks_with_pp_last:
            new_group = torch.distributed.new_group(
                ranks=pp_ranks
            )
            if world_rank in pp_ranks:
                pp_pg = new_group

    return {
        "world_pg": world_pg,
        "world_rank_matrix": ranks,
        "dp_pg": dp_pg,
        "tp_pg": tp_pg,
        "pp_pg": pp_pg
    }

def main():
    initialize_torch_distributed()
    groups = get_process_groups(
        data_parallel_size=1,
        tensor_parallel_size=2,
        pipeline_parallel_size=2
    )
    world_pg = groups["world_pg"]
    world_size = world_pg.size()

    for i in range(world_size):
        if world_pg.rank() == i:
            print(f"###### RANK {i} / {world_size}")
            results = ["    "]

            dp_pg = groups["dp_pg"]
            if dp_pg is not None:
                results.append(f"dp_rank={dp_pg.rank()}/{dp_pg.size()} | ")

            tp_pg = groups["tp_pg"]
            if tp_pg is not None:
                results.append(f"tp_rank={tp_pg.rank()}/{tp_pg.size()} | ")

            pp_pg = groups["pp_pg"]
            if pp_pg is not None:
                results.append(f"pp_rank={pp_pg.rank()}/{pp_pg.size()} | ")

            print("".join(results))
        dist.barrier(world_pg)

if __name__ == "__main__":
    main()