import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image
from PIL import Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
local_image_path = "images/poppy-with-outfit.png"  # <-- change this to your local image path
init_image = Image.open(local_image_path).convert("RGB")

prompt = "Puppy in a room full of toys, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)

output_path = "output-new.png"
image.save(output_path)
print(f"Generated image saved at {output_path}")