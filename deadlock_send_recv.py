import os

import torch
from torch import distributed as dist
from torch.distributed import ProcessGroup


def init_distributed():
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    backend = "gloo"

    # Call the init process.
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return True

def exercise2(group: ProcessGroup):
    # Exercise: Communicate tensor0 to rank 1 and tensor1 to rank 1
    if group.rank() == 0:
        # tensor0 is define in rank 0
        tensor0 = torch.randn(3,10, dtype=torch.float)
    elif group.rank() == 1:
        tensor1 = torch.randn(4,11, dtype=torch.float)

    # assign buffers:
    if group.rank() == 0:
        tensor1 = torch.empty(4,11, dtype=torch.float)
    else:
        tensor0 = torch.empty(3, 10, dtype=torch.float)

    # send data
    print("Sending")
    if group.rank() == 0:
        dist.send(tensor0, dst=1, group=group)
    else:
        dist.send(tensor1, dst=0, group=group)

    # recv data
    print("Receiving")
    if group.rank() == 0:
        dist.recv(tensor1, src=1, group=group)
    else:
        dist.recv(tensor0, src=0, group=group)

    # This deadlocks

    raise NotImplemented("TODO")

def main():
    init_distributed()
    group = dist.group.WORLD

    exercise2(group)

if __name__ == "__main__":
    main()