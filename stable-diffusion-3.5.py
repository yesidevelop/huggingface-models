import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "Full-body portrait of a small anthropomorphic puppy with a humanoid body, standing upright in a straight, full frontal pose facing the camera. Cute and playful appearance with big expressive eyes, floppy ears, soft fur, and childlike proportions. Mischievous yet innocent facial expression. Wearing a neat, casual child-style outfit such as a t-shirt and shorts. Bright, evenly distributed studio lighting, no shadows. Pure white background with no environment elements. High-resolution, sharp focus, realistic textures, family-friendly character design, capturing charm, innocence, and playful personality.",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("capybara.png")
