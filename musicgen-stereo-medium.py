from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "Playful, whimsical, and upbeat orchestral music, "
    "with lively piano, pizzicato strings, marimba, "
    "light percussion, cartoonish, fun, cinematic, "
    "animated movie soundtrack style"
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 800})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
