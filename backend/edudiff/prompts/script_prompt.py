from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an expert mathematics teacher.

Your task is to solve mathematical problems step by step
and explain them clearly as if teaching a student.

Guidelines:
- Explain every important step in simple language
- Show mathematical expressions where needed
- Handle ALL types of math problems
- Be clear and student-friendly

Output format:
1. Restate the problem briefly
2. Step-by-step solution with explanations
3. Final answer or conclusion
"""

def generate_math_solution(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content
