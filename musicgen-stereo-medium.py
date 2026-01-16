from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "Cinematic orchestral score, soft tremolo strings, haunting solo cello, minimal piano notes with high reverb, subtle low-frequency pulses, suspenseful atmosphere, mysterious and slow-paced, high quality, 48kHz."
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 800})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
