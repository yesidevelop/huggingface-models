import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from accelerate import init_empty_weights, dispatch_model

# pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

with init_empty_weights():
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16
    )

pipeline = dispatch_model(pipeline, device_map="auto")  


# --- Enable memory-saving features ---
pipeline.enable_attention_slicing()
pipeline.vae.enable_tiling()
pipeline.enable_model_cpu_offload()

# pipeline.to('cuda')
image1 = Image.open("images/poppy.png")
image2 = Image.open("images/dada.png")
prompt = "The big dog in martial arts suit is on the left, the puppy with skyblue suit is on the right, facing each other in the central park square."
inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 20,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_plus.png")
    print("image saved at", os.path.abspath("output_image_edit_plus.png"))
