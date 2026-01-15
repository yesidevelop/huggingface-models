from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "A single continuous cinematic instrumental composition that evolves smoothly from beginning to end with no cuts, no pauses, and no abrupt transitions; opening with a slow, gentle, and mysterious mood in a minor key at approximately 55 BPM using soft ambient pads, distant strings, and light piano textures; gradually increasing tension in the middle section through subtle dissonant harmonies, deeper string drones, low pulsing bass, and restrained rhythmic movement while maintaining smooth continuity and rising anticipation; final section resolving into a warm, uplifting, and hopeful mood by slowly shifting toward a major key feel, introducing brighter harmonies, gentle melodic strings, and soft orchestral warmth; dynamics rise naturally and then settle without climax spikes; no breakdowns, no silence, no tempo jumps; cohesive thematic identity throughout; smooth looping ending; instrumental only; no vocals, no lyrics; cinematic, emotional, and uniquely atmospheric."
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 800})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
