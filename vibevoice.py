import torch
import soundfile as sf
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/VibeVoice-1.5B",
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    "microsoft/VibeVoice-1.5B",
    trust_remote_code=True
).to(device)

text = """
The forest was silent, except for the distant sound of wind moving through the trees.
"""

inputs = processor(text=text, return_tensors="pt").to(device)

with torch.no_grad():
    audio = model.generate(**inputs)

sf.write("output.wav", audio.cpu().numpy(), 24000)
