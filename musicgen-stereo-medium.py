from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "Energetic and catchy K-pop music, "
    "bright synths, punchy electronic drums, groovy bassline, "
    "playful vocal melodies, layered harmonies, sparkling arpeggios, "
    "upbeat, cheerful, vibrant, fun, danceable, polished pop production, "
    "cinematic and colorful, high energy performance vibe, "
    "modern Korean pop soundtrack style"
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 800})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
