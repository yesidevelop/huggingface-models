from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "A cheerful, upbeat song where the main melody is played by a cat meowing repeatedly, pop music style"
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 800})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
