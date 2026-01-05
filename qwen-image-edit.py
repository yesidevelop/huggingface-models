import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open("images/poppy.png")
# image2 = Image.open("images/dada-real.png")
prompt = "Full-room view of a living room, showing a small puppy playing dangerously with real fire on the floor. The room has a TV mounted on the wall, a grey sofa, a white rug, and two grey chairs. A chandelier hangs from the ceiling. Shelves display toys including a teddy bear and puzzles. Scattered blocks and a blue toy truck are on the floor. The scene is dynamic and chaotic, emphasizing the danger of the fire interacting with the puppy. Realistic lighting and textures for furniture, rug, toys, and fire, full-room perspective showing all elements clearly, high-resolution, realistic style"
inputs = {
    # "image": [image1, image2],
    "image": [image1],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_plus.png")
    print("image saved at", os.path.abspath("output_image_edit_plus.png"))
