import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query, evidence):
    context = "\n\n".join(
        f"[{i+1}] {e['content']}"
        for i, e in enumerate(evidence)
    )

    prompt = f"""
Answer strictly using the evidence below.
If evidence is insufficient, say so clearly.

Evidence:
{context}

Question: {query}
"""

    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
