# make sure you're logged in with `huggingface-cli login`
from contextlib import nullcontext

from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, PNDMScheduler
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import datetime
import time
import torch

import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# torch._C._jit_set_nvfuser_single_node_mode(True)

# torch.backends.cudnn.benchmark = False
torch.manual_seed(12315)

if torch.cuda.is_available():
    torch.cuda.manual_seed(12315)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    scheduler=scheduler,
    use_auth_token=True
).to(device)


# unet_traced = torch.jit.load("unet_traced_CL_nofloat_singlefuse_noscale.pt")
# unet_traced.in_channels = pipe.unet.in_channels
# pipe.unet = unet_traced


prompt = "a photo of an astronaut riding a horse on mars"

# warmup
with torch.inference_mode():
    image = pipe([prompt]*1, num_inference_steps=8).images[0]

for _ in range(3):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        image = pipe([prompt]*1, num_inference_steps=50).images[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Pipeline inference took {time.time() - start_time:.2f} seconds")
image.save(f"pics/pic_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_pt_fp16_tracedunet_CL_nofloat_singlefuse_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
#         ) as prof:
with nullcontext():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        image = pipe([prompt]*1, num_inference_steps=8).images[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Pipeline inference took (w/ Profiler) {time.time() - start_time:.2f} seconds")