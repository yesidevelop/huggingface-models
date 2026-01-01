import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

# Model ID
model_id = "hunyuanvideo-community/HunyuanVideo"

# Load transformer in half precision (float16) for GPU
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.float16  # use float16 for GPU
)

# Load pipeline
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16
)

# Move entire pipeline to GPU
pipe = pipe.to("cuda")

# Memory optimizations
pipe.vae.enable_tiling()          # reduce VAE memory usage
pipe.enable_attention_slicing()   # reduce attention memory usage

# Video generation
output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
).frames[0]

# Export video
export_to_video(output, "output.mp4", fps=15)

print("Video saved as output.mp4")
