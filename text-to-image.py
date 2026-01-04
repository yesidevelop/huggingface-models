import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")

local_image_path = "images/poppy-with-outfit.png"  # <-- change this to your local image path
init_image = Image.open(local_image_path).convert("RGB")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, image=init_image).images


# Save the first generated image
image[0].save("astronaut_on_horse.png")