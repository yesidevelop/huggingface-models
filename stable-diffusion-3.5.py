import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "A full-body, front-facing portrait of the father of the dog character, shown in a realistic humanoid bodybuilder form. Brown and white fur with a distinct white blaze running down the forehead and muzzle, clearly conveying a strong paternal resemblance. The character stands approximately 6 feet tall with a powerful, athletic build and moderately defined bodybuilder muscles—broad chest, thick arms, strong legs, and a solid, imposing posture. He is wearing professional bodybuilder attire, including fitted bodybuilding shorts and wrist wraps, with no shirt, fully showcasing his muscular upper body. His expression is serious, calm, and protective, radiating strength, discipline, and authority. Realistic anatomy, natural muscle definition (not exaggerated), detailed fur on the face only, clean humanoid skin on the body, studio-style cinematic lighting, sharp focus, ultra-high realism, high-resolution, neutral background.",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("capybara.png")
