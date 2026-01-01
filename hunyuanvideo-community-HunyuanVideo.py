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
    prompt="A high-quality cinematic shot of a serene female yoga instructor in a bright, minimalist studio. She is wearing a vibrant pink athletic yoga set. The camera starts in a medium shot as she gracefully transitions into a Tree Pose (Vrikshasana), lifting one foot to her inner thigh and bringing her hands to a prayer position. Soft natural light flows through large windows in the background. She maintains perfect balance on one leg, her core visible and engaged, with a calm and focused expression.",
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
).frames[0]
export_to_video(output, "output.mp4", fps=15)
