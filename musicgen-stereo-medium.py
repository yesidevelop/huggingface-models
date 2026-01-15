from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "A high-energy, loud funky instrumental track with a strong, infectious groove; tempo around 105–115 BPM; tight electric bass with prominent slap and syncopated patterns; rhythmic clean electric guitar playing muted funk strums; punchy brass stabs used sparingly; steady drum kit with crisp hi-hats and snare backbeat; bright analog synth accents; consistent rhythm throughout; no breakdowns, no tempo changes; upbeat, playful, and confident mood; clean modern production; instrumental only; suitable as a continuous background groove with a smooth looping structure."
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 800})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
