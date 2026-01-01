from moviepy import VideoFileClip, AudioFileClip

# 1. Load the video and audio files
video = VideoFileClip("output.mp4")
audio = AudioFileClip("musicgen_out.wav")

# 2. Set the audio of the video clip
# This replaces the original audio (if any)
final_video = video.with_audio(audio)

# 3. Write the result to a file
# You can use 'mp4', 'mkv', etc.
final_video.write_videofile("merged_output.mp4", codec="libx264", audio_codec="aac")

# 4. Close the clips to free up system resources
video.close()
audio.close()