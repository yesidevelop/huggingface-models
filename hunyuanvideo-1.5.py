import torch
from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video

# ----------------------------
# Configuration
# ----------------------------
dtype = torch.bfloat16
device = "cuda:0"
FPS = 15
SECONDS = 3
NUM_FRAMES = FPS * SECONDS  # 75 frames

seed = 42

prompt = (
    "A cinematic wildlife documentary scene of a powerful tiger hunting a deer "
    "in a dense green forest, fast dynamic movement, realistic animals, "
    "dramatic lighting, shallow depth of field, ultra realistic, 4k nature documentary"
)

negative_prompt = (
    "cartoon, anime, low quality, blurry, artifacts, distorted anatomy, "
    "extra limbs, unrealistic motion"
)

# prompt = (
#     "A cinematic K-pop music video featuring a mystical daemon performing "
#     "intense dance moves on a futuristic neon-lit stage, dynamic camera angles, "
#     "stylish costumes, dramatic lighting, vibrant colors, ultra realistic, 4k cinematic"
# )

# negative_prompt = (
#     "cartoon, anime, low quality, blurry, artifacts, distorted anatomy, "
#     "extra limbs, unrealistic motion, messy background, bad lighting"
# )



# ----------------------------
# Load pipeline
# ----------------------------
pipe = HunyuanVideo15Pipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    torch_dtype=dtype
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.enable_attention_slicing()

# ----------------------------
# Generator
# ----------------------------
generator = torch.Generator(device=device).manual_seed(seed)

# ----------------------------
# Generate video
# ----------------------------
video = pipe(
    prompt=prompt,
    generator=generator,
    num_frames=NUM_FRAMES,
    num_inference_steps=40,
).frames[0]

# ----------------------------
# Export
# ----------------------------
export_to_video(video, "tiger_hunting_deer_10s.mp4", fps=FPS)

print("Video saved: tiger_hunting_deer_10s.mp4")
