from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

music = synthesiser("Calm ambient yoga music, soft pads, slow tempo, peaceful, cinematic, relaxing atmosphere", forward_params={"do_sample": True})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
