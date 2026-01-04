import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
from diffusers.models import IPAdapter
from PIL import Image

device = "cuda"

# Load base SDXL (NOT the refiner)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
).to(device)

pipe.enable_xformers_memory_efficient_attention()

# Load IP-Adapter
ip_adapter = IPAdapter(
    pipe,
    "IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.safetensors",
    device=device
)

# Load reference image (your puppy)
reference_image = Image.open("images/poppy-with-outfit.png").convert("RGB")

prompt = (
    "Wide-angle full-room interior scene. "
    "A large, bright, well-decorated playroom fully visible from wall to wall and floor to ceiling. "
    "Many toys scattered across the room: toy cars, balls, plush animal toys. "
    "TV lounge style room, colorful walls with framed scenery pictures. "
    "Soft daylight, high illumination. "
    "The puppy stands naturally in the room."
)

negative_prompt = (
    "close-up, cropped, zoomed in, partial room, face only, blurry, "
    "distorted anatomy, extra limbs, duplicate character"
)

# Generate image
image = ip_adapter.generate(
    pil_image=reference_image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    scale=0.75,          # identity strength (0.6–0.85 recommended)
    width=1024,
    height=1024,
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42
)[0]

image.save("output_ip_adapter_room.png")
print("Saved: output_ip_adapter_room.png")
