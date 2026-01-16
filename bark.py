from transformers import pipeline
import numpy as np
import scipy.io.wavfile as wavfile

synthesiser = pipeline("text-to-speech", "suno/bark-small")

speech = synthesiser(
    "Hello, my dog is cooler than you!",
    forward_params={"do_sample": True}
)

audio = speech["audio"]
sampling_rate = speech["sampling_rate"]

# Ensure numpy array
audio = np.asarray(audio)

# Normalize to int16 PCM
audio = audio / np.max(np.abs(audio))
audio_int16 = (audio * 32767).astype(np.int16)

wavfile.write("bark_out.wav", sampling_rate, audio_int16)