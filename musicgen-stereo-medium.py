from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "Cinematic fairy tale soundtrack, slow and ethereal, ethereal chime harp arpeggios, soaring emotional solo violin melody, magical atmosphere, lush reverb, 60 BPM, mystical and enchanting."
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 1600})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
