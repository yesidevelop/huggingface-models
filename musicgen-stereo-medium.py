from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

music = synthesiser("Calm ambient yoga music with breathy bamboo flute melody, soft evolving synth pads, slow tempo, no percussion, peaceful, spiritual, spacious, cinematic", forward_params={"do_sample": True, "max_new_tokens": 128})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
