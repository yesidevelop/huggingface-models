import torch
import soundfile as sf
from transformers import pipeline

synthesiser = pipeline("text-to-audio", "facebook/musicgen-stereo-medium", device="cuda:0", torch_dtype=torch.float16)

music = synthesiser("Calm ambient yoga music, soft pads, slow tempo, peaceful, cinematic, relaxing atmosphere", forward_params={"max_new_tokens": 256})

sf.write("musicgen_out.wav", music["audio"][0].T, music["sampling_rate"])