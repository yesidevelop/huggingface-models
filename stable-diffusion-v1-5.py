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

prompt = "A small, cute puppy sitting in a bright, cheerful living room. The room is full of colorful toys scattered on the floor: balls, cars, and stuffed animals. There is a TV lounge with a sofa and a coffee table. Walls are decorated with framed pictures of landscapes and cute illustrations. Sunlight coming through large windows, soft shadows, high-resolution, realistic lighting, cozy and inviting atmosphere, cinematic perspective showing the whole room, warm and vibrant colors."

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)

output_path = "output.png"
image.save(output_path)
print(f"Generated image saved at {output_path}")