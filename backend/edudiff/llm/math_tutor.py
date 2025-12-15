import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from ..prompts.tutor_prompt import SYSTEM_PROMPT


load_dotenv()


def _get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Create and return a configured OpenAI client.

    The API key is read from the OPENAI_API_KEY environment variable
    unless an explicit key is provided.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please configure it before using the math tutor."
        )
    return OpenAI(api_key=key)

client = _get_openai_client()


def generate_math_solution(question: str) -> str:
    """
    Generate a full, step-by-step mathematical solution and explanation
    for the given question using the OpenAI Chat Completions API.

    The response mimics a ChatGPT-style math tutor:
    - Restates the problem
    - Shows step-by-step reasoning
    - Ends with a clear final answer or conclusion
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ],
        temperature=0.35,
    )

    content = response.choices[0].message.content
    return content.strip() if content else ""


