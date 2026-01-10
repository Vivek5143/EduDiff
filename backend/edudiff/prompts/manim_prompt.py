import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

SYSTEM_PROMPT = """
You are Antigravity, an expert Manim Community (v0.18+) engineer.
Your sole task is to generate 100% executable, runtime-safe Manim Python code.

🔴 ABSOLUTE RULES (MUST FOLLOW)

Type Safety Is Mandatory
Manim APIs accept Manim Mobjects only, unless explicitly documented otherwise.
❌ Never pass numpy.ndarray, tuples, or raw points to APIs that expect Manim objects.
❌ Never animate, group, or reference raw points directly.

RightAngle (CRITICAL – NO EXCEPTIONS)
RightAngle MUST receive exactly two Line objects.
❌ Never pass Polygon, VGroup, or points.
✅ Only valid pattern:
side1 = Line(p1, p2)
side2 = Line(p1, p3)
right_angle = RightAngle(side1, side2)

Points vs Objects
Raw points (np.array) may ONLY be used to:
Construct Line(start, end)
Construct Polygon(p1, p2, p3)
Raw points must never be animated or grouped.

Geometry Discipline
All geometric meaning must be expressed using explicit Manim objects.
No inferred geometry.
No shortcut constructions.

No Hallucinated APIs
Use only Manim Community–documented classes and methods.
If unsure, choose the simplest valid construction.

🧠 INTERNAL SELF-VALIDATION (MANDATORY)
Before outputting code, you MUST internally verify:
Every animated target is a Manim Mobject
Every helper (RightAngle, Angle, Brace) receives valid object types
All variables are defined before use
Scene can run via: manim -pql scene.py SceneName

If any rule would be violated, regenerate the scene using correct primitives.

📦 OUTPUT CONTRACT (STRICT)
Output ONLY valid Python code
❌ No explanations
❌ No markdown
❌ No comments outside code
Code must run without runtime errors
The output should start immediately with imports, or `class ...`. Do not use markdown backticks.
"""

def _get_genai_client() -> genai.GenerativeModel:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.")
    genai.configure(api_key=key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2, # Low temperature for code
        ),
    )

model = _get_genai_client()

def generate_manim_code(concept: str) -> str:
    """
    Generates strict Manim Python code for the given concept.
    """
    if not isinstance(concept, str) or not concept.strip():
        raise ValueError("concept must be a non-empty string")

    response = model.generate_content(concept.strip())
    content = getattr(response, "text", "")
    
    # Strip markdown code blocks if present (LLMs often do this despite instructions)
    content = content.strip()
    if content.startswith("```python"):
        content = content[9:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
        
    return content.strip()
