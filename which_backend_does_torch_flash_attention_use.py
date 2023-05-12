import torch
from torch.nn import functional as F

# Lets define a helpful benchmarking function:
import torch.utils.benchmark as benchmark
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

# Lets define the hyper-parameters of our input
batch_size = 32
max_sequence_len = 1024
num_heads = 32
embed_dimension = 32

dtype = torch.float16
device="cuda"

kwargs = {
    "query": torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype),
    "key": torch.rand(batch_size, 1, max_sequence_len, embed_dimension, device=device, dtype=dtype).expand(batch_size, num_heads, max_sequence_len, embed_dimension),
    "value": torch.rand(batch_size, 1, max_sequence_len, embed_dimension, device=device, dtype=dtype).expand(batch_size, num_heads, max_sequence_len, embed_dimension),
    "attn_mask": (torch.randn(batch_size, 1, max_sequence_len, max_sequence_len, device=device, dtype=dtype) > 0).expand(batch_size, num_heads, max_sequence_len, max_sequence_len),
    "dropout_p": 0.1,
    "is_causal": False,
}


print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, **kwargs):.3f} microseconds")

# Lets explore the speed of each of the 3 implementations
from torch.backends.cuda import sdp_kernel, SDPBackend

# Helpful arguments mapper
backend_map = {
    SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
}

with sdp_kernel(**backend_map[SDPBackend.MATH]):
    print(f"The math implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, **kwargs):.3f} microseconds")


with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
    try:
        print(f"The flash attention implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, **kwargs):.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")

with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
    try:
        print(f"The memory efficient implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, **kwargs):.3f} microseconds")
    except RuntimeError:
        print("EfficientAttention is not supported. See warnings for reasons.")
