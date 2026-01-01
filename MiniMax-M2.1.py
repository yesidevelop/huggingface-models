# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="MiniMaxAI/MiniMax-M2.1", trust_remote_code=True)
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)