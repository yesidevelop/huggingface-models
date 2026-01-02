from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "Whimsical and magical Pied Piper style music, "
    "playful melody led by pan flute and piccolo, "
    "light harp and woodwinds, fast tempo, "
    "airy, cheerful, mischievous, fairy-tale soundtrack, "
    "cinematic and enchanting, lively folk-style rhythm"
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 800})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
