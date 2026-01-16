from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "Dark pop instrumental, steady 808 heartbeat kick drum, crisp closed hi-hats, soft suspenseful synth drones, muted plucked guitar, minor key, 95 BPM, atmospheric and tense, rhythmic but quiet."
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 3200})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
