"""
High-level math tutoring pipeline.

This module exposes generate_math_steps(), which wraps the LLM-backed
math tutor and converts its free-form explanation into a simple
structure used by downstream components (e.g., Manim rendering).
"""

from typing import Dict, List

from ..llm.math_tutor import generate_math_solution


def _split_into_lines(text: str) -> List[str]:
    """Split a block of text into non-empty, trimmed lines."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def generate_math_steps(question: str) -> Dict[str, object]:
    """
    Generate structured math explanation steps for a given question.

    This function:
    - Calls the ChatGPT-like math tutor to get a full, step-by-step explanation.
    - Splits the explanation into a list of lines for downstream rendering.
    - Returns a dictionary compatible with the rest of the EduDiff pipeline:

      {
        "steps": [list of explanation lines],
        "explanation": "summary string"
      }

    Note: We do NOT parse or "solve" any math ourselves. All mathematical
    reasoning is delegated to the LLM.
    """
    full_text = generate_math_solution(question)

    # Basic, model-agnostic structuring: treat each non-empty line as a step.
    lines = _split_into_lines(full_text)

    # Use all lines as "steps" for the EquationTransform scene.
    steps: List[str] = lines if lines else [full_text] if full_text else []

    # For the textual explanation/summary, use the last few lines joined together,
    # which usually includes the conclusion and final answer.
    if len(lines) >= 3:
        summary_lines = lines[-3:]
    else:
        summary_lines = lines or [full_text]

    explanation = "\n".join(summary_lines) if summary_lines else ""

    return {
        "steps": steps,
        "explanation": explanation,
    }


