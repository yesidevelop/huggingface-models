import scipy
from transformers import AutoProcessor, BarkModel
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small")
inputs = processor(""" Hello, my dog is cooler than you! """, voice_preset="v2/en_speaker_5")
audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()
sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav",rate=sample_rate, data=audio_array)