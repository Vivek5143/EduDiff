import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

VOICE_SYSTEM_PROMPT = """
You are an expert educational narration writer.

You are given Manim Python code.
Your task is to generate a spoken narration script that matches the visuals exactly.

🔴 ABSOLUTE RULES (NON-NEGOTIABLE)

Manim Code Is the Source of Truth
You may ONLY describe objects, equations, and steps that explicitly exist in the provided Manim code.
❌ Do NOT invent visuals, steps, or equations.

Visual Synchronization
Narration must follow the logical order of animations in construct().
Use short, clear sentences suitable for TTS.

No Visual Assumptions
❌ Do not say “as you can see” unless the object is explicitly created.
❌ Do not describe colors, positions, or highlights unless present in code.

Educational Tone
Simple
Calm
Neutral
Student-friendly

🧠 SCRIPT STRUCTURE (MANDATORY)
Output narration as pure JSON:
{
  "title": "Video title",
  "segments": [
    {
      "start_after_animation": 0,
      "text": "Narration text here"
    }
  ]
}

start_after_animation refers to animation index order in Manim (0-indexed).
0 means after the first self.play/self.add.
No timestamps.
No markdown.
No explanations.

📦 OUTPUT CONTRACT (STRICT)
Output ONLY valid JSON.
No comments.
No markdown.
No emojis.
No extra text.
"""

def _get_genai_client() -> genai.GenerativeModel:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.")
    genai.configure(api_key=key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=VOICE_SYSTEM_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )

model = _get_genai_client()

def generate_voice_script(manim_code: str) -> dict:
    """
    Generates a synchronized voice script for the given Manim code.
    Returns a dict with 'title' and 'segments'.
    """
    if not isinstance(manim_code, str) or not manim_code.strip():
        raise ValueError("manim_code must be a non-empty string")

    response = model.generate_content(manim_code.strip())
    content = getattr(response, "text", "")
    
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError as e:
        # Strict enforcement: Fail loud if not valid JSON
        raise ValueError(f"Voice generation failed: Output was not valid JSON. Error: {e}\nContent: {content}") from e
