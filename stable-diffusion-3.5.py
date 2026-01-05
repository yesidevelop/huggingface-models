import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "A full-body, front-facing portrait of an adult male anthropomorphic dog, based on a Chihuahua x Australian Shepherd mix, depicted in a highly realistic humanoid form. Medium-length brown and white fur with a distinct white blaze running down the forehead and muzzle. Semi-floppy ears, proportional adult canine facial structure, masculine features including a strong jawline, defined cheek structure, slightly narrower eyes, and a longer, mature muzzle. No baby-like proportions. The character is tall (approximately 6 feet), with a powerful, athletic build, broad shoulders, thick arms, and realistic, natural muscle definition. Confident, protective posture with a calm, authoritative expression. He wears simple, well-fitted adult clothing: a fitted short-sleeve t-shirt that emphasizes the chest and arms, tailored dark trousers, leather belt, and clean leather shoes. Clothing appears realistic with natural fabric texture and folds. Ultra-realistic anatomy, detailed fur texture on the face, ears, and arms. Cinematic studio lighting, soft shadows, sharp focus, high detail, ultra-high resolution. Neutral light gray background. No cartoon style, no chibi proportions, no exaggerated eyes, no childish facial features.",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("capybara.png")
