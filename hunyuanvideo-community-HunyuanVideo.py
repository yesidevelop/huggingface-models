import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)

# Enable memory savings
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="Cinematic video of a female yoga instructor in a pink outfit performing Tree Pose. She stands in a sunlit, minimalist studio, balancing steadily on one leg with hands in prayer position. Sharp focus on her engaged core and calm expression. Smooth camera zoom, realistic movement, serene atmosphere.",
    height=720,
    width=1280,
    num_frames=151,
    num_inference_steps=30,
).frames[0]
export_to_video(output, "output.mp4", fps=15)

print(f'Number of frames: {len(output)}')

