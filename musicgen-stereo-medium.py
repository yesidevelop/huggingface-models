import torch
from transformers import pipeline
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

music_pipe = pipeline(
    "text-to-audio",
    model="facebook/musicgen-stereo-medium",
    device=0 if device == "cuda" else -1
)

music_prompt = (
    "Calm ambient yoga music, soft pads, slow tempo, "
    "peaceful, cinematic, relaxing atmosphere"
)

audio = music_pipe(
    music_prompt,
    forward_params={
        "max_new_tokens": 512  # controls length
    }
)

sf.write(
    "bg_music.wav",
    audio["audio"][0],
    samplerate=audio["sampling_rate"]
)
