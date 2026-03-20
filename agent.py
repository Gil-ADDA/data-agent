from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()

SYSTEM_PROMPT = """
You are a data collection AI agent.
Your job is to collect information from the internet based on user requests.
When you don't have tools yet — explain clearly what you WOULD do step by step.
"""

def run_agent(user_request: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request}
        ]
    )
    return response.choices[0].message.content

# Test run
result = run_agent("Collect the top 5 news headlines from a news website")
print("--- Agent Response ---")
print(result)
