import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid
from PIL import Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

local_image_path = "images/poppy-with-outfit.png"  # <-- change this to your local image path
init_image = Image.open(local_image_path).convert("RGB")

prompt = "Zoomed-out full view of a bright, cheerful playroom with an anthropomorphic puppy, Poppy, playing energetically. The entire room is visible, showing a cozy TV lounge, colorful furniture, and well-decorated walls with framed pictures of scenic landscapes. Toys are scattered everywhere: toy cars, balls, stuffed animals, building blocks, and other playful items covering the floor and shelves. Poppy is mid-action, interacting with toys, mischievous and cute expression. Bright, high-lit, warm lighting, vibrant colors, cartoonish yet detailed textures, family-friendly style, cinematic perspective capturing the full layout of the room, playful chaos, and lively atmosphere."

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)