import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open("/home/ubuntu/huggingface-models/images/poppy.png")
image2 = Image.open("/home/ubuntu/huggingface-models/images/dada-real.png")
image3 = Image.open("/home/ubuntu/huggingface-models/images/last_frame.jpg")
prompt = "A tall 6 feet tall dada in the martial arts suit, looking angry and worried as he scolds to small 2 feet kid poppy who plays with fire. His expression shows frustration and concern because a fire has caught the sofa. Puppy looks sad and scared, ears slightly drooped, eyes wide with guilt and fear. The camera is zoomed in on Poppy, capturing her emotional reaction to the dangerous situation. The room is visible with a burning sofa, subtle smoke and light reflections, realistic fire effects. Cinematic lighting, high-resolution, natural textures, dynamic shadows, realistic style, no cartoonish elements. Motion is natural and fluid, emphasizing the urgency and tension of the scene."
inputs = {
    "image": [image1, image2, image3],
    # "image": [image1],
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
