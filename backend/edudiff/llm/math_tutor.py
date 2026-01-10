import os
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai

from ..prompts.tutor_prompt import SYSTEM_PROMPT


load_dotenv()


def _get_genai_model(api_key: Optional[str] = None) -> genai.GenerativeModel:
    """
    Configure and return a Gemini model client.

    Reads the API key from GOOGLE_API_KEY unless an explicit key is provided.
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please configure it before using the math tutor."
        )

    genai.configure(api_key=key)

    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.35,
            response_mime_type="text/plain",
        ),
    )


model = _get_genai_model()


def generate_math_solution(question: str) -> str:
    """
    Generate a full, step-by-step mathematical explanation for the question.

    The model is instructed to:
    - Restate the problem
    - Explain each step clearly in plain text (no LaTeX, no special symbols)
    - Provide a final answer/conclusion
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    response = model.generate_content(question.strip())

    content = getattr(response, "text", None)
    return content.strip() if content else ""
