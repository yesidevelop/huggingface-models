from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "[Instrumental] A slow, cinematic fairy tale theme. Begins with a delicate chime harp and celesta duet. A soulful solo violin enters with a sweeping melody, supported by an ethereal female choir and light flute trills. Orchestral, magical, 60 BPM, high-fidelity, soaring and emotional."
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 1600})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
