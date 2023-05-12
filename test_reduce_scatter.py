import os

import torch
from torch import distributed as dist, nn


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
    group = dist.distributed_c10d._get_default_group()
    batch_size_sharded = 5
    batch_size_unsharded = batch_size_sharded * group.size()
    in_hidden_size_sharded = 2
    out_hidden_size = 3

    sharded_input = torch.randn(batch_size_unsharded, in_hidden_size_sharded, device="cuda")

    model = nn.Linear(in_hidden_size_sharded, out_hidden_size, device="cuda")

    out = model(sharded_input) # [batch_size_unsharded, out_hidden_size]

    # All reduce
    real_out_all_reduce = torch.empty(batch_size_unsharded, out_hidden_size, device="cuda")
    real_out_all_reduce.copy_(out)
    dist.all_reduce(real_out_all_reduce, op=dist.ReduceOp.SUM, group=group)

    # Reduce scatter + all_gather
    sharded_real_out = torch.empty(batch_size_sharded, out_hidden_size, device="cuda")
    dist.reduce_scatter_tensor(sharded_real_out, out, group=group, op=dist.ReduceOp.SUM)

    expected_shape = [batch_size_sharded, out_hidden_size]
    assert tuple(sharded_real_out.shape) == tuple(expected_shape), f"Expected {expected_shape}, got {sharded_real_out.shape}"
    torch.testing.assert_close(
        real_out_all_reduce[group.rank() * batch_size_sharded: (group.rank() + 1) * batch_size_sharded],
        sharded_real_out
    )

    real_out_reduce_scatter = torch.empty(batch_size_unsharded, out_hidden_size, device="cuda")
    dist.all_gather_into_tensor(real_out_reduce_scatter, sharded_real_out, group=group)

    torch.testing.assert_close(real_out_all_reduce, real_out_reduce_scatter)


if __name__ == "__main__":
    main()