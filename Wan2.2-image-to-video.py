import torch
import numpy as np
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# ----------------------------
# Configuration
# ----------------------------
model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
dtype = torch.bfloat16

# Load pipeline with automatic multi-GPU support
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",        # Automatically spread model across GPUs
    low_cpu_mem_usage=True
)

# Enable memory optimizations
pipe.enable_model_cpu_offload()  # Offload unused weights to CPU
pipe.vae.enable_tiling()         # Reduce memory usage in VAE
pipe.enable_attention_slicing()  # Save VRAM during attention

# ----------------------------
# Prepare input image
# ----------------------------
image = load_image(
    "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-560w,f_auto,q_auto:best/rockcms/2025-07/250709-Kpop-Demon-Hunters-vl-256p-ea5850.jpg"
)

max_area = 480 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]

height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

# ----------------------------
# Prompts and generation settings
# ----------------------------
prompt = (
    "A dynamic concert scene of three animated female K-pop idols performing "
    "on a neon-lit stage with dramatic fantasy lighting, energetic choreography, "
    "and modern futuristic costumes inspired by Korean pop culture, vibrant colors, "
    "dramatic motion blur, anime-influenced stylized 3D animation, detailed expressions "
    "and synchronized dance poses, high energy performance with musical instruments and light effects, "
    "cinematic composition, ultra-detailed, sharp focus, hype crowd in background"
)

negative_prompt = (
    "stiff posing, static figures, unnatural anatomy, extra limbs, demonic weapons, "
    "combat poses, motion blur artifacts, low resolution, smudged faces, text or logos, "
    "dull lighting, muted colors, visual noise, broken limbs, closed eyes, awkward expressions, "
    "busy background that distracts from the performance"
)

generator = torch.Generator().manual_seed(0)

# ----------------------------
# Generate video frames
# ----------------------------
output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=81,
    guidance_scale=3.5,
    num_inference_steps=40,
    generator=generator
).frames[0]

# ----------------------------
# Export video
# ----------------------------
export_to_video(output, "i2v_output.mp4", fps=16)

print("Video saved as i2v_output.mp4")
