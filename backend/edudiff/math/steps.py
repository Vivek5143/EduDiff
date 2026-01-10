"""
High-level math tutoring pipeline.

This module exposes generate_math_steps(), which wraps the LLM-backed
math tutor and converts its free-form explanation into a simple
structure used by downstream components (e.g., Manim rendering).
"""

from typing import Dict, List

from ..llm.math_tutor import generate_math_solution


def _split_into_lines(text: str) -> List[str]:
    """
    Split text into logical lines for animation/narration.

    Preference order:
    - Existing newlines from the LLM (common for step lists).
    - Fallback to splitting sentences by period if no newlines present.
    """
    raw = text.strip()
    if not raw:
        return []

    if "\n" in raw:
        parts = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        parts = [seg.strip() for seg in raw.split(".") if seg.strip()]

    return parts


def generate_math_steps(question: str) -> Dict[str, object]:
    """
    Generate structured math explanation steps for a given question using Gemini.

    - Delegates all reasoning to the LLM (no hardcoded math).
    - Returns plain-text steps and a concise explanation summary.
    """
    full_text = generate_math_solution(question)

    steps = _split_into_lines(full_text)
    if not steps and full_text:
        steps = [full_text.strip()]

    explanation = full_text.strip() if full_text else ""

    return {
        "steps": steps,
        "explanation": explanation,
    }


