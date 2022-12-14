import time

import torch
torch.backends.cuda.matmul.allow_tf32 = True

"""
The goal is to see if there's a possibility we can somehow achieve 250 TFLOPs/s using pytorch;
source: https://mobile.twitter.com/IrwanBello/status/1599881662560899073
"""

def main():
    if not torch.cuda.is_available():
        raise ValueError("We're truing to run the benchmark on pytorch")
    tensorboard_folder = "tb_logs"

    dtype = torch.bfloat16
    device = torch.device("cuda")

    # max_g_per_gpu = 70 * (2 ** 30) # play it safe assume we have 70G
    num_of_runs = 20000

    K = 4096
    M = 2304
    N = 4608

    # constraints to be respected
    # Source: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
    # A100 tensorcore constraints
    assert K % 64 == 0
    assert M % 64 == 0
    assert N % 64 == 0
    # wave quantization: assuming tiles are 256 x 128
    assert M % 256 == 0
    assert N % 128 == 0
    assert (M // 256) * (N // 128) % 108 == 0

    dummy_input = torch.rand(M, K, dtype=dtype, device=device)
    weight = torch.rand(K, N, dtype=dtype, device=device)

    # warmup
    for _ in range(10):
        out = dummy_input @ weight
        del out

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_folder),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ):
        out = dummy_input @ weight
        del out
    for _ in range(num_of_runs - 1):
        out = dummy_input @ weight
        del out
    torch.cuda.synchronize()
    duration = time.time() - start_time

    total_number_of_flops = 2 * K * M * N * num_of_runs / 10 ** 12
    print(f"Seconds: {time.time() - start_time} s")
    print(f"TFLOPs/s: {total_number_of_flops / duration} TFLOP/s")


if __name__ == "__main__":
    main()
