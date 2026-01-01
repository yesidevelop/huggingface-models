# merge_video_audio.py
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

# Paths to your generated files
video_path = "output.mp4"       # from file1.py
audio_path = "musicgen_out.wav" # from file2.py
output_path = "final_output.mp4"

# Load video and audio
video = VideoFileClip(video_path).subclip(0, 10)  # take first 10 seconds
audio = AudioFileClip(audio_path).subclip(0, 10)  # first 10 seconds of audio

# Set audio to video
video = video.set_audio(audio)

# Write final output
video.write_videofile(output_path, fps=15, codec="libx264", audio_codec="aac")
