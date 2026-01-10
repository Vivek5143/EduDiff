import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()


def _get_genai_client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=key)


client = _get_genai_client()


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
    """Legacy helper using Gemini; kept for backward compatibility."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    prompt = f"{SYSTEM_PROMPT}\n\nStudent question:\n{question.strip()}"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.4,
        ),
    )
    content = getattr(response, "text", None)
    return content.strip() if content else ""
