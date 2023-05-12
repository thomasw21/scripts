import os

import torch
from torch import nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel


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
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return True

def comm_hook(state, bucket):
    buffer = bucket.buffer()
    print(buffer.shape)
    return dist.all_reduce(buffer, op=dist.ReduceOp.SUM, async_op=True).get_future().then(lambda fut: buffer)

def main():
    initialize_torch_distributed()

    model = nn.Sequential(nn.Linear(3,3), nn.ReLU())

    # Sync weights
    dist.all_reduce(model[0].weight)
    dist.all_reduce(model[0].bias)

    # DDP
    dist._DEFAULT_FIRST_BUCKET_BYTES = 2
    model = DistributedDataParallel(
        model,
        bucket_cap_mb=2**-19,
        find_unused_parameters=False
    )
    print(model.broadcast_bucket_size, model.bucket_bytes_cap)
    model.register_comm_hook(None, comm_hook)

    n_iter = 3
    losses = []
    for i in range(n_iter):
        input = torch.randn(2,3)
        # print(input)
        # This syncs even if we add the context later
        with model.no_sync():
            loss = model(input).sum()
            # loss = model(input).sum()
            pass

        losses.append(loss)

    for loss in losses:
        with model.no_sync():
            loss.backward()

    # Check that model are synced
    group = dist.distributed_c10d._get_default_group()
    for name, param in model.named_parameters():
        reference_rank = 0
        if reference_rank == group.rank():
            reference_tensor = param.grad
        else:
            reference_tensor = torch.empty_like(param)
        dist.broadcast(reference_tensor, src=reference_rank)
        # print(param.grad)
        torch.testing.assert_close(reference_tensor, param.grad, atol=0, rtol=0)
    print("Done")

if __name__ == "__main__":
    main()