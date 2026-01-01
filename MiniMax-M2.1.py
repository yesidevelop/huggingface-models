from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="MiniMaxAI/MiniMax-M2.1",
    trust_remote_code=True,
    device_map="cuda"   # <-- This moves the model to GPU
)

messages = [
    {"role": "user", "content": "Who are you?"},
]

output = pipe(messages)
print(output)
