import os

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

def main():
    initialize_torch_distributed()

    w = torch.randn(1,3, device="cuda")
    dist.all_reduce(w, op=dist.ReduceOp.AVG)

    print(f"Sucess: {dist.get_rank()}/{dist.get_world_size()}")


if __name__ == "__main__":
    main()