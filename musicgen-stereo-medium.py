from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "A slow, cinematic dark-fantasy background score with a haunting, emotional tone; minor key; steady tempo around 60–70 BPM; soft sustained strings and low cello drones as the foundation; distant female choir pads used sparingly for atmosphere, not melody; subtle piano notes with long reverb; light ambient textures like wind, breath, and room tone; no strong rhythm or percussion; no abrupt transitions; smooth looping structure; restrained dynamics; melancholic, mysterious, and somber mood; suitable as continuous background music for a dark fairy-tale scene involving a quiet child and an ominous witch; instrumental only; no vocals, no lyrics; cinematic, minimal, and unobtrusive."
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 800})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
