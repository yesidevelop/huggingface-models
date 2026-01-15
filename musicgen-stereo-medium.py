from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")

prompt = (
    "A playful, upbeat children’s song featuring cute cat ‘meow’ vocalizations used rhythmically as hooks, bouncy melody, simple joyful harmony, bright and cheerful mood, medium-fast tempo, light percussion, soft synths and bells, cartoon-like and innocent, loop-friendly, wholesome and fun, suitable for a kids animation or cute animal video."
)
music = synthesiser(prompt, forward_params={"do_sample": True, "max_new_tokens": 800})

scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
