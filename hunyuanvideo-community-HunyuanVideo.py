import torch
import numpy as np
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

# Model ID
model_id = "hunyuanvideo-community/HunyuanVideo"

# Load transformer in half precision (float16) for GPU
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.float16
)

# Load pipeline
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16
)

# Move pipeline to GPU
pipe = pipe.to("cuda")

# Memory optimizations
pipe.vae.enable_tiling()          # reduce VAE memory usage
pipe.enable_attention_slicing()   # reduce attention memory usage
pipe.enable_model_cpu_offload()   # offload parts of model to CPU to save GPU memory

# Video generation in chunks to fit GPU memory
num_frames = 61
chunk_size = 10  # generate 10 frames at a time
all_frames = []

# for start in range(0, num_frames, chunk_size):
#     end = min(start + chunk_size, num_frames)
#     frames = pipe(
#         prompt="A cat walks on the grass, realistic",
#         height=320,
#         width=512,
#         num_frames=end-start,
#         num_inference_steps=30
#     ).frames
    
#     # Convert PIL Images to NumPy arrays
#     frames = [np.array(frame) for frame in frames]
#     all_frames.extend(frames)

output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=320,
    width=512,
    num_frames=1,
    num_inference_steps=30,
).frames[0]
# Export full video
# export_to_video(all_frames, "output.mp4", fps=15)
export_to_video(output, "output.mp4", fps=15)

print("Video saved as output.mp4")
