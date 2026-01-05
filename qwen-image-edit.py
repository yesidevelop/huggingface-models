import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open("images/poppy.png")
image2 = Image.open("images/dada.png")
prompt = "Full-body scene in Central Park square: Dada, a big dog wearing a martial arts suit, 6 feet tall, on the left, looking at Poppy with a gentle warning, as if saying “don’t break the toys”; Poppy, a small puppy wearing a sky-blue suit, 2 feet tall, on the right, playfully interacting with toys such as a red rubber ball, a small frisbee, a squeaky bone, and a plush teddy bear. Highly realistic textures for fur and clothing, natural lighting, dynamic expressions and posture, high-resolution, photorealistic rendering, detailed environment of the park square."
inputs = {
    "image": [image1, image2],
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
