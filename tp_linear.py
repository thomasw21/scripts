import os
from typing import Optional

import torch
from torch import nn
from torch import distributed as dist
from torch.distributed import ProcessGroup

class DifferentiableIdentity(torch.autograd.Function):
    """All-reduce gradients in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return DifferentiableAllReduceSum.apply(grad_output, group), None


class DifferentiableAllReduceSum(torch.autograd.Function):
    """All-reduce in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def differentiable_identity(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableIdentity.apply(tensor, group)


def differentiable_all_reduce_sum(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableAllReduceSum.apply(tensor, group)


class TensorParallelColumnLinear(nn.Linear):
    """Takes un-sharded input and return sharded output"""
    def __init__(
        self,
        in_features,
        out_features,
        pg: dist.ProcessGroup,
        bias=True,
        device=None,
        dtype=None,
    ):
        self.pg = pg
        self.world_size = pg.size()

        assert out_features % self.world_size == 0

        self.in_features = in_features
        self.out_features = out_features // self.world_size

        super().__init__(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = differentiable_identity(x, group=self.pg)
        return super().forward(x)

    def extra_repr(self) -> str:
        return f"tp_rank={self.pg.rank()}, {super().extra_repr()}, unsharded_out_features={self.out_features * self.world_size}"


class TensorParallelRowLinear(nn.Linear):
    """Takes sharded input and return un-sharded output"""
    def __init__(
        self,
        in_features,
        out_features,
        pg: dist.ProcessGroup,
        bias=True,
        device=None,
        dtype=None,
    ):
        self.pg = pg
        self.world_size = pg.size()

        assert in_features % self.world_size == 0

        self.in_features = in_features // self.world_size
        self.out_features = out_features

        # No need to shard the bias term, only rank 0 would have it
        bias = self.pg.rank() == 0 and bias

        super().__init__(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return differentiable_all_reduce_sum(out, group=self.pg)




def initialize_torch_distributed():
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    assert torch.cuda.is_available()
    # Set the device id.
    # TODO @thomasw21: `torch.cuda.device_count` should return the number of device on a single node. We assume the nodes to be homogeneous (same number of gpus per node)
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    backend = "nccl"

    # Call the init process.
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return True

def test_column_linear(group: ProcessGroup):
    """Test TPColumnLinear: Input in un sharded, ie duplicated across all ranks"""
    device=torch.device("cuda")
    batch_size = 7
    in_features = 3
    out_features_per_rank = 5
    out_features = out_features_per_rank * group.size()
    model = TensorParallelColumnLinear(in_features=in_features, out_features=out_features, pg=group, device=device)
    dummy_input = torch.randn(batch_size, in_features, device=device)
    # synchronize all inputs across ranks
    torch.distributed.all_reduce(dummy_input, op=torch.distributed.ReduceOp.AVG, group=group)
    output = model(dummy_input)
    # TODO: Do whatever you want with output

def test_row_linear(group: ProcessGroup):
    """Test TPRowLinear: Input in sharded, and output is synchronized"""

    device=torch.device("cuda")
    batch_size = 7
    in_features_per_rank = 3
    in_features = in_features_per_rank * group.size()
    out_features = 5
    model = TensorParallelRowLinear(in_features=in_features, out_features=out_features, pg=group, device=device)
    dummy_input = torch.randn(batch_size, in_features_per_rank, device=device)
    output = model(dummy_input)

    # Check it's synchronized
    reference_rank = 0
    if group.rank() == reference_rank:
        reference_output = output
    else:
        reference_output = torch.empty_like(output)
    torch.distributed.broadcast(reference_output, src=torch.distributed.get_global_rank(group=group, group_rank=reference_rank), group=group)
    torch.testing.assert_close(reference_output, output, atol=0, rtol=0)

    # TODO: Do whatever you want with output

def main():
    initialize_torch_distributed()
    group = torch.distributed.distributed_c10d._get_default_group()

    test_column_linear(group=group)
    test_row_linear(group=group)
    print("Done")

if __name__ == "__main__":
    main()