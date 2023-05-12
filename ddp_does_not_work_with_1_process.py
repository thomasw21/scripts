import os

from torch import distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel


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
        init_method=init_method
    )
    return True

def main():
    init_distributed()
    group = dist.group.WORLD

    model = nn.Linear(3,10)

    model = DistributedDataParallel(model, process_group=group)
    print("Done")

if __name__ == "__main__":
    main()