from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "Whimsical fairy-tale music with delicate chimes, music box bells, soft harp, and warm ambient textures. Innocent, joyful, and magical feeling, like a bedtime story."
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 1600})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
