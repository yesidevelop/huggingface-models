from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "Gentle orchestral fairy-tale music with a magical and storybook atmosphere. Soft string ensemble playing warm sustained chords, delicate harp arpeggios, and a celesta adding sparkling accents. A light flute carries the main melody, supported by subtle piano notes. Tempo is slow to moderate, flowing and calm. Mood is whimsical, innocent, and enchanting, suitable for a children’s fairy tale narration. No drums, no modern instruments, purely classical fantasy orchestration."
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 2000})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
