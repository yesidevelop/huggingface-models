import torch
from transformers import CsmForConditionalGeneration, AutoProcessor

model_id = "sesame/csm-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs
text = "[0]That is actually hilarious, haha! I can't believe it." # `[0]` for speaker id 0
inputs = processor(text, add_special_tokens=True).to(device)

# another equivalent way to prepare the inputs
conversation = [
    {"role": "0", "content": [{"type": "text", "text": "That is actually hilarious, haha! I can't believe it."}]},
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(device)

# infer the model
audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, "example_without_context.wav")
