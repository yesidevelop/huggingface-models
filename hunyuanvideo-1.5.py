import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

# ----------------------------
# Configuration
# ----------------------------
MODEL_ID = "tencent/HunyuanVideo-1.5"
OUTPUT_VIDEO = "tiger_hunting_deer_10s.mp4"

FPS = 24
SECONDS = 10
NUM_FRAMES = FPS * SECONDS  # 240 frames = 10 seconds

PROMPT = (
    "A cinematic wildlife documentary scene of a powerful tiger hunting a deer "
    "in a dense green forest, dynamic movement, realistic animals, dramatic lighting, "
    "shallow depth of field, ultra realistic, 4k nature documentary style"
)

NEGATIVE_PROMPT = (
    "cartoon, anime, low quality, blurry, artifacts, distorted anatomy, "
    "extra limbs, unrealistic motion"
)

# ----------------------------
# Load model
# ----------------------------
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
)

pipe = HunyuanVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

# Optional memory optimizations
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# ----------------------------
# Generate video
# ----------------------------
with torch.autocast("cuda", dtype=torch.float16):
    video_frames = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_frames=NUM_FRAMES,
        height=720,
        width=1280,
        guidance_scale=7.0,
        num_inference_steps=30,
    ).frames

# ----------------------------
# Export to MP4
# ----------------------------
export_to_video(video_frames, OUTPUT_VIDEO, fps=FPS)

print(f"Video saved as {OUTPUT_VIDEO}")
