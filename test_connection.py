from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()

message = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Say hello in one sentence"}
    ]
)

print("Connection successful!")
print("Model response:", message.choices[0].message.content)
