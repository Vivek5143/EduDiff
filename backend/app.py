from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import os
import sys
import tempfile
import subprocess
import logging
import uuid
import shutil
import json
import warnings

# Suppress pydub RuntimeWarning about ffmpeg (emitted during manim import)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*ffmpeg.*')

from manim import *
from google import genai
from dotenv import load_dotenv
from datetime import datetime
import time
import random
import io
import re

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, 
    static_url_path='/static',
    static_folder='static')
CORS(app)

app.logger.setLevel(logging.INFO)

# Configure Manim
config.media_dir = "media"
config.video_dir = "videos"
config.images_dir = "images"
config.text_dir = "texts"
config.tex_dir = "tex"
config.log_dir = "log"
config.renderer = "cairo"
config.text_renderer = "cairo"
config.use_opengl_renderer = False

# Set up required directories
def setup_directories():
    """Create all required directories for the application"""
    directories = [
        os.path.join(app.root_path, 'static'),
        os.path.join(app.root_path, 'static', 'videos'),
        os.path.join(app.root_path, 'tmp'),
        os.path.join(app.root_path, 'media'),
        os.path.join(app.root_path, 'media', 'videos'),
        os.path.join(app.root_path, 'media', 'videos', 'scene'),
        os.path.join(app.root_path, 'media', 'videos', 'scene', '720p30'),
        os.path.join(app.root_path, 'media', 'videos', 'scene', '1080p60')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        app.logger.info(f'Created directory: {directory}')

# Set up directories at startup
setup_directories()

# Ensure static directory exists
os.makedirs(os.path.join(app.root_path, 'static', 'videos'), exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- GenAI / rendering defaults ---------------------------------------------
GENAI_MODEL = os.getenv('GENAI_MODEL', 'gemini-2.5-flash')
RENDER_QUALITY_DEFAULT = os.getenv('RENDER_QUALITY', 'low').lower()

# Initialize GenAI (new google-genai SDK)
genai_client = None
try:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        genai_client = genai.Client(api_key=api_key)
        logger.info(f"GenAI client initialized with model: {GENAI_MODEL}")
    else:
        logger.warning("No GOOGLE_API_KEY or GEMINI_API_KEY found. AI features will be disabled.")
except Exception as e:
    logger.error(f"Failed to initialize GenAI client: {e}")

# Set media and temporary directories with fallback to local paths
if os.environ.get('DOCKER_ENV'):
    app.config['MEDIA_DIR'] = os.getenv('MEDIA_DIR', '/app/media')
    app.config['TEMP_DIR'] = os.getenv('TEMP_DIR', '/app/tmp')
else:
    app.config['MEDIA_DIR'] = os.path.join(os.path.dirname(__file__), 'media')
    app.config['TEMP_DIR'] = os.path.join(os.path.dirname(__file__), 'tmp')

# Ensure directories exist
os.makedirs(app.config['MEDIA_DIR'], exist_ok=True)
os.makedirs(app.config['TEMP_DIR'], exist_ok=True)
os.makedirs(os.path.join(app.config['MEDIA_DIR'], 'videos', 'scene', '720p30'), exist_ok=True)
os.makedirs(os.path.join(app.static_folder, 'videos'), exist_ok=True)


def sanitize_input(text):
    """Sanitize input text by removing extra whitespace and newlines"""
    return ' '.join(text.strip().split())

def sanitize_title(text):
    """Sanitize text for use in title"""
    text = sanitize_input(text)
    return text.replace('"', '').replace("'", "").strip()

# --- LaTeX helpers -----------------------------------------------------------
LATEX_COMMAND_HINTS = [
    r"\\frac", r"\\sum", r"\\int", r"\\sqrt", r"\\alpha", r"\\beta",
    r"\\pi", r"\\sin", r"\\cos", r"\\tan", r"\\left", r"\\right",
]

def is_likely_latex(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if any(d in t for d in ["$$", "$", r"\\(", r"\\)", r"\\[", r"\\]"]):
        return True
    if any(cmd in t for cmd in LATEX_COMMAND_HINTS):
        return True
    if ("^" in t or "_" in t) and " " not in t.strip()[:3]:
        return True
    return False

def clean_latex(text: str) -> str:
    t = text.strip()
    # remove common delimiters
    t = re.sub(r"^\$+|\$+$", "", t)
    t = re.sub(r"^\\\(|\\\)$", "", t)
    t = re.sub(r"^\\\[|\\\]$", "", t)
    return t.strip()

def generate_latex_scene_code(expr: str) -> str:
    expr = clean_latex(expr)
    return f"""from manim import *\n\nclass MainScene(Scene):\n    def construct(self):\n        title = Title('LaTeX')\n        eq = MathTex(r"{expr}").scale(1.2)\n        self.play(Write(title))\n        self.play(Write(eq))\n        self.wait()\n"""

# --- AI helpers --------------------------------------------------------------

def extract_text(response) -> str:
    """
    Extract text from LLM response, handling both string content and structured content blocks.
    
    Args:
        response: LLM response object (Gemini API response)
        
    Returns:
        str: Extracted text content, or empty string if none found
    """
    if not response:
        logger.warning("extract_text: response is None or empty")
        return ""
    
    # Log raw response for debugging (once) - use INFO level so it's visible
    try:
        logger.info(f"Raw LLM response type: {type(response)}")
        logger.info(f"Raw LLM response has attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        
        # Try to log the actual structure
        if hasattr(response, 'candidates'):
            logger.info(f"Response has candidates: {response.candidates is not None}")
            if response.candidates:
                logger.info(f"Number of candidates: {len(response.candidates)}")
                if len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    logger.info(f"First candidate type: {type(candidate)}")
                    logger.info(f"First candidate attributes: {[attr for attr in dir(candidate) if not attr.startswith('_')]}")
                    if hasattr(candidate, 'content'):
                        logger.info(f"Candidate content type: {type(candidate.content)}")
                        if hasattr(candidate.content, 'parts'):
                            logger.info(f"Content parts type: {type(candidate.content.parts)}, length: {len(candidate.content.parts) if candidate.content.parts else 0}")
                            if candidate.content.parts:
                                logger.info(f"First part type: {type(candidate.content.parts[0])}")
                                logger.info(f"First part: {candidate.content.parts[0]}")
    except Exception as e:
        logger.error(f"Error logging raw response: {e}", exc_info=True)
    
    # Try direct text attribute first (Gemini API simple case)
    # Also try calling it as a method if it's callable
    if hasattr(response, 'text'):
        try:
            text_attr = response.text
            # If text is a property/method, try calling it
            if callable(text_attr):
                text_attr = text_attr()
            if text_attr and isinstance(text_attr, str) and text_attr.strip():
                logger.info("Extracted text from response.text attribute")
                return text_attr.strip()
        except Exception as e:
            logger.error(f"Error accessing response.text: {e}")
            # Also log prompt feedback if available
            if hasattr(response, 'prompt_feedback'):
                logger.error(f"Prompt feedback: {response.prompt_feedback}")
    
    # Try response.text as a method call (new Gemini SDK)
    try:
        if hasattr(response, 'text') and callable(getattr(response, 'text', None)):
            text_result = response.text()
            if text_result and isinstance(text_result, str) and text_result.strip():
                logger.info("Extracted text from response.text() method")
                return text_result.strip()
    except Exception as e:
        logger.debug(f"Error calling response.text(): {e}")
    
    # Try Gemini API structured format: candidates[0].content.parts (most common for Gemini)
    if hasattr(response, 'candidates'):
        try:
            candidates = response.candidates
            if candidates and len(candidates) > 0:
                candidate = candidates[0]
                if hasattr(candidate, 'content'):
                    content = candidate.content
                    if content:
                        # Try to get parts
                        parts = None
                        if hasattr(content, 'parts'):
                            parts = content.parts
                        elif hasattr(content, 'get') and callable(content.get):
                            # If content is dict-like
                            parts = content.get('parts')
                        
                        if parts:
                            text_parts = []
                            for part in parts:
                                # Handle different part types
                                part_text = None
                                
                                # Try as object with text attribute
                                if hasattr(part, 'text'):
                                    part_text = part.text
                                # Try as dict
                                elif isinstance(part, dict):
                                    part_text = part.get('text') or part.get('output_text')
                                # Try as string
                                elif isinstance(part, str):
                                    part_text = part
                                
                                if part_text:
                                    text_parts.append(str(part_text))
                            
                            if text_parts:
                                result = '\n'.join(text_parts).strip()
                                logger.info(f"Extracted text from candidates[0].content.parts ({len(text_parts)} parts, {len(result)} chars)")
                                return result
        except (AttributeError, IndexError, KeyError, TypeError) as e:
            logger.error(f"Error accessing candidates[0].content.parts: {e}", exc_info=True)
    
    # Try message.content if it exists (OpenAI-style format)
    if hasattr(response, 'choices') and response.choices:
        try:
            message = response.choices[0].message
            if hasattr(message, 'content'):
                content = message.content
                # If content is a non-empty string, return it
                if isinstance(content, str) and content.strip():
                    logger.info("Extracted text from choices[0].message.content (string)")
                    return content.strip()
                # If content is a list, extract text from blocks
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            # Check for OpenAI-style blocks
                            if block.get('type') == 'text' and 'text' in block:
                                text_parts.append(str(block['text']))
                            # Check for output_text type
                            elif block.get('type') == 'output_text' and 'text' in block:
                                text_parts.append(str(block['text']))
                        elif hasattr(block, 'text'):
                            text_parts.append(str(block.text))
                    if text_parts:
                        result = '\n'.join(text_parts).strip()
                        logger.info(f"Extracted text from choices[0].message.content (list, {len(text_parts)} blocks)")
                        return result
        except (AttributeError, IndexError, KeyError, TypeError) as e:
            logger.debug(f"Error accessing choices[0].message.content: {e}")
    
    # Last resort: try to convert response to string or use __str__
    try:
        if hasattr(response, '__str__'):
            str_repr = str(response)
            if str_repr and str_repr.strip() and str_repr != str(type(response)):
                logger.warning("Extracted text using __str__ fallback (may not be accurate)")
                return str_repr.strip()
    except Exception as e:
        logger.debug(f"Error in __str__ fallback: {e}")
    
    # Log warning if no text found
    logger.error("extract_text: No text found in response using any extraction method")
    logger.error(f"Response type: {type(response)}, Response repr: {repr(response)[:500]}")
    return ""

def extract_code_from_response(text: str) -> str:
    if not text:
        return ""
    # Try fenced code blocks with language
    m = re.search(r"```(?:python)?\n([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def sanitize_manim_code(code: str) -> str:
    """
    Sanitize Gemini-generated Manim code:
    - Remove markdown code fences
    - Strip leading explanation text
    - Start from 'from manim import' line
    """
    if not code:
        return ""
    
    # Remove markdown code fences if present
    code = re.sub(r'^```(?:python)?\s*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
    
    # Find the line containing "from manim import"
    lines = code.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        if 'from manim import' in line.lower():
            start_idx = i
            break
    
    # Extract code starting from 'from manim import'
    sanitized = '\n'.join(lines[start_idx:]).strip()
    
    return sanitized


def generate_ai_manim_code(concept: str) -> str:
    if genai_client is None:
        return ""
    try:
        # Backend guard: Detect equation-based questions
        concept_lower = concept.lower()
        equation_keywords = ["solve", "=", "equation", "find x", "find y"]
        is_equation = any(keyword in concept_lower for keyword in equation_keywords)
        
        # Use the strict prompt (it already handles equation detection, but we log it)
        full_prompt = generate_manim_prompt(concept)
        
        if is_equation:
            logger.info(f"Detected equation-solving question: {concept}")
        
        resp = genai_client.models.generate_content(
            model=GENAI_MODEL,
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,  # Lower temperature for more deterministic output
            ),
        )
        
        # Extract text using helper function (logging happens inside extract_text)
        content = extract_text(resp)
        
        # Validate extracted content is not empty
        if not content or not content.strip():
            logger.error("LLM returned empty output")
            raise ValueError("LLM returned empty output")
        
        code = extract_code_from_response(content)
        
        # Sanitize the code: remove markdown, extract from 'from manim import'
        code = sanitize_manim_code(code)
        
        # Validate sanitized code is not empty
        if not code or not code.strip():
            logger.error("Extracted code is empty after sanitization")
            raise ValueError("LLM returned empty output")
        
        return code
    except Exception as e:
        logger.error(f"AI generation failed: {e}")
        return ""

def generate_explanation(concept):
    """Generate a short text explanation of the concept."""
    if genai_client is None:
        return f"Here is a visual explanation of {concept}."
    try:
        prompt = (
            "You are a helpful math tutor. Provide a concise, 2-sentence explanation "
            "of the requested concept. Do not use LaTeX formatting, just plain text.\n\n"
            f"Concept: {concept}"
        )
        resp = genai_client.models.generate_content(
            model=GENAI_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.7,
            ),
        )
        
        # Extract text using helper function
        text = extract_text(resp)
        return text if text else f"Explanation of {concept}."
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        return f"Here is a visual explanation of {concept}."

def generate_manim_prompt(concept):
    """Generate a strict, deterministic prompt for Gemini to create Manim code"""
    # Detect if this is an equation-solving problem
    concept_lower = concept.lower()
    is_equation = any(keyword in concept_lower for keyword in ["solve", "=", "equation", "find x", "find y"])
    
    equation_context = ""
    if is_equation:
        equation_context = """
Question Type: LINEAR_EQUATION

CRITICAL: This is an equation-solving problem. You MUST:
- Show step-by-step algebraic transformations
- Use MathTex for each equation step
- Use TransformMatchingTex to animate between steps
- NEVER use Axes, NumberPlane, plot(), or any graph/coordinate system
- Display the final answer clearly

Example structure for "Solve for x: 3x - 5 = 10":
1. Show: 3x - 5 = 10
2. Transform to: 3x = 15
3. Transform to: x = 5
4. Highlight the answer

"""
    
    return f"""You are a deterministic Manim code generator. You are NOT a creative AI. You are a code compiler.

Your ONLY task: Generate EXECUTABLE Python Manim code that solves the math problem step-by-step.

==============================
MANDATORY RULES (NO EXCEPTIONS)
==============================

1. OUTPUT ONLY PYTHON CODE. No markdown, no explanations, no comments outside code.
2. DO NOT generate graphs or axes for equation solving.
3. DO NOT use Axes, NumberPlane, or plot() for equations.
4. DO NOT use generic template animations.
5. You are not a creative AI. You are a deterministic code generator.
6. Every step MUST be shown using MathTex and TransformMatchingTex.
7. Add self.wait(1) after every transformation.
8. End with self.wait(0.5) to finish.

==============================
REQUIRED SCENE STRUCTURE
==============================

from manim import *

class MainScene(Scene):
    def construct(self):
        # Step 1: Show original equation
        eq1 = MathTex(r"<original_equation>")
        self.play(Write(eq1))
        self.wait(1)
        
        # Step 2: Transform to next step
        eq2 = MathTex(r"<next_equation>")
        self.play(TransformMatchingTex(eq1, eq2))
        self.wait(1)
        
        # Continue for all steps...
        
        # Final answer highlight
        answer_box = SurroundingRectangle(eq_final, color=GREEN)
        self.play(Create(answer_box))
        self.wait(1)
        self.wait(0.5)

==============================
FORBIDDEN PATTERNS (HARD REJECT)
==============================

Your code will be REJECTED if it contains:
- Axes
- NumberPlane
- plot(
- GraphScene
- begin_ambient_camera_rotation
- while True
- self.wait() without duration

==============================
{equation_context}==============================
INPUT PROBLEM
==============================

Solve the following problem visually using Manim:

{concept}
"""

def select_template(concept):
    """Select appropriate template based on the concept."""
    concept = concept.lower().strip()

    # CRITICAL: Bypass templates if the user wants to SOLVE an equation
    solve_indicators = ['solve', '=', 'calculate', 'simplify', 'find', 'step-by-step']
    if any(ind in concept for ind in solve_indicators):
        return None
    
    # Define template mappings with keywords
    template_mappings = {
        'pythagorean': {
            'keywords': ['pythagoras', 'pythagorean', 'right triangle', 'hypotenuse'],
            'generator': generate_pythagorean_code
        },
        'quadratic': {
            'keywords': ['quadratic', 'parabola', 'x squared', 'x^2'],
            'generator': generate_quadratic_code
        },
        'trigonometry': {
            'keywords': ['sine', 'cosine', 'trigonometry', 'trig', 'unit circle'],
            'generator': generate_trig_code
        },
        '3d_surface': {
            'keywords': ['3d surface', 'surface plot', '3d plot', 'three dimensional'],
            'generator': generate_3d_surface_code
        },
        'sphere': {
            'keywords': ['sphere', 'ball', 'spherical'],
            'generator': generate_sphere_code
        },
        'cube': {
            'keywords': ['cube', 'cubic', 'box'],
            'generator': generate_cube_code
        },
        'derivative': {
            'keywords': ['derivative', 'differentiation', 'slope', 'rate of change'],
            'generator': generate_derivative_code
        },
        'integral': {
            'keywords': ['integration', 'integral', 'area under curve', 'antiderivative'],
            'generator': generate_integral_code
        },
        'matrix': {
            'keywords': ['matrix', 'matrices', 'linear transformation'],
            'generator': generate_matrix_code
        },
        'eigenvalue': {
            'keywords': ['eigenvalue', 'eigenvector', 'characteristic'],
            'generator': generate_eigenvalue_code
        },
        'complex': {
            'keywords': ['complex', 'imaginary', 'complex plane'],
            'generator': generate_complex_code
        },
        'differential_equation': {
            'keywords': ['differential equation', 'ode', 'pde'],
            'generator': generate_diff_eq_code
        }
    }
    
    # Find best matching template
    best_match = None
    max_matches = 0
    
    for template_name, template_info in template_mappings.items():
        matches = sum(1 for keyword in template_info['keywords'] if keyword in concept)
        if matches > max_matches:
            max_matches = matches
            best_match = template_info['generator']
    
    # Return best matching template
    if best_match and max_matches > 0:
        try:
            return best_match()
        except Exception as e:
            logger.error(f"Error generating template {best_match.__name__}: {str(e)}")
            return None
    
    # Default to None to trigger AI generation
    return None

def generate_pythagorean_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create triangle
        triangle = Polygon(
            ORIGIN, RIGHT*3, UP*4,
            color=WHITE
        )
        
        # Add labels using Text instead of MathTex
        a = Text("a", font_size=36).next_to(triangle, DOWN)
        b = Text("b", font_size=36).next_to(triangle, RIGHT)
        c = Text("c", font_size=36).next_to(
            triangle.get_center() + UP + RIGHT,
            UP+RIGHT
        )
        
        # Add equation using MathTex
        equation = MathTex(r"a^2 + b^2 = c^2").scale(1.1)
        equation.to_edge(UP)
        
        # Create the animation
        self.play(Create(triangle))
        self.play(Write(a), Write(b), Write(c))
        self.play(Write(equation))
        self.wait()'''

def generate_derivative_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create coordinate system
        axes = Axes(
            x_range=[-2, 2],
            y_range=[-1, 2],
            axis_config={"include_tip": True}
        )
        
        # Add custom labels
        x_label = Text("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(axes.y_axis.get_end(), UP)
        
        # Create function
        def func(x):
            return x**2
            
        graph = axes.plot(func, color=BLUE)
        
        # Create derivative function
        def deriv(x):
            return 2*x
            
        derivative = axes.plot(deriv, color=RED)
        
        # Create labels
        func_label = Text("f(x) = x²").set_color(BLUE)
        deriv_label = Text("f'(x) = 2x").set_color(RED)
        
        # Position labels
        func_label.to_corner(UL)
        deriv_label.next_to(func_label, DOWN)
        
        # Create animations
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(graph), Write(func_label))
        self.wait()
        self.play(Create(derivative), Write(deriv_label))
        self.wait()'''

def generate_integral_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create coordinate system
        axes = Axes(
            x_range=[-2, 2],
            y_range=[-1, 2],
            axis_config={"include_tip": True}
        )
        
        # Add custom labels
        x_label = Text("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(axes.y_axis.get_end(), UP)
        
        # Create function
        def func(x):
            return x**2
            
        graph = axes.plot(func, color=BLUE)
        
        # Create area
        area = axes.get_area(
            graph,
            x_range=[0, 1],
            color=YELLOW,
            opacity=0.3
        )
        
        # Create labels
        func_label = Text("f(x) = x²").set_color(BLUE)
        integral_label = Text("Area = 1/3").set_color(YELLOW)
        
        # Position labels
        func_label.to_corner(UL)
        integral_label.next_to(func_label, DOWN)
        
        # Create animations
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(graph), Write(func_label))
        self.wait()
        self.play(FadeIn(area), Write(integral_label))
        self.wait()'''

def generate_3d_surface_code():
    return '''from manim import *
import numpy as np

class MainScene(ThreeDScene):
    def construct(self):
        # Set up the axes with better spacing
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-2, 2, 0.5],
            x_length=6,
            y_length=6,
            z_length=4,
            axis_config={"include_tip": True}
        )
        
        # Create surface function
        def param_surface(u, v):
            x = u
            y = v
            z = np.sin(np.sqrt(x**2 + y**2))
            return np.array([x, y, z])
        
        # Create surface with optimized resolution
        surface = Surface(
            lambda u, v: param_surface(u, v),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(20, 20),
            should_make_jagged=False,
            stroke_opacity=0
        )
        
        # Add color and styling
        surface.set_style(
            fill_opacity=0.8,
            stroke_color=BLUE,
            stroke_width=0.5,
            fill_color=BLUE
        )
        surface.set_fill_by_value(
            axes=axes,
            colors=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)],
            axis=2
        )
        
        # Set up the scene
        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=45 * DEGREES,
            zoom=0.6
        )
        
        # Animate
        self.begin_ambient_camera_rotation(rate=0.2)
        self.play(Create(axes))
        self.play(Create(surface))
        self.wait(2)
        self.stop_ambient_camera_rotation()
'''

def generate_sphere_code():
    return '''from manim import *
import numpy as np

class MainScene(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        axes = ThreeDAxes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            z_range=[-3, 3],
            x_length=6,
            y_length=6,
            z_length=6
        )
        
        # Create sphere
        radius = 2
        sphere = Surface(
            lambda u, v: np.array([
                radius * np.cos(u) * np.cos(v),
                radius * np.cos(u) * np.sin(v),
                radius * np.sin(u)
            ]),
            u_range=[-PI/2, PI/2],
            v_range=[0, TAU],
            checkerboard_colors=[BLUE_D, BLUE_E],
            resolution=(15, 32)
        )
        
        # Create radius line and label
        radius_line = Line3D(
            start=ORIGIN,
            end=[radius, 0, 0],
            color=YELLOW
        )
        r_label = Text("r", font_size=36).set_color(YELLOW)
        r_label.rotate(PI/2, RIGHT)
        r_label.next_to(radius_line, UP)
        
        # Create volume formula
        volume_formula = MathTex(r"\frac{4}{3}\pi r^3").to_corner(UL)
        
        # Add everything to scene
        self.add(axes)
        self.play(Create(sphere))
        self.wait()
        self.play(Create(radius_line), Write(r_label))
        self.wait()
        self.play(Write(volume_formula))
        self.wait()
        
        # Rotate camera for better view
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()'''

def generate_cube_code():
    return '''from manim import *

class MainScene(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        axes = ThreeDAxes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            z_range=[-3, 3]
        )
        
        # Create cube
        cube = Cube(side_length=2, fill_opacity=0.7, stroke_width=2)
        cube.set_color(BLUE)
        
        # Labels for sides
        a_label = Text("a", font_size=36).set_color(YELLOW)
        a_label.next_to(cube, RIGHT)
        
        # Surface area formula
        area_formula = Text(
            "A = 6a^2"
        ).to_corner(UL)
        
        # Add everything to scene
        self.add(axes)
        self.play(Create(cube))
        self.wait()
        self.play(Write(a_label))
        self.wait()
        self.play(Write(area_formula))
        self.wait()
        
        # Rotate camera for better view
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()'''

def generate_matrix_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create matrices
        matrix_a = VGroup(
            Text("2  1"),
            Text("1  3")
        ).arrange(DOWN)
        matrix_a.add(SurroundingRectangle(matrix_a))
        
        matrix_b = VGroup(
            Text("1"),
            Text("2")
        ).arrange(DOWN)
        matrix_b.add(SurroundingRectangle(matrix_b))
        
        # Create multiplication symbol and equals sign
        times = Text("×")
        equals = Text("=")
        
        # Create result matrix
        result = VGroup(
            Text("4"),
            Text("7")
        ).arrange(DOWN)
        result.add(SurroundingRectangle(result))
        
        # Position everything
        equation = VGroup(
            matrix_a, times, matrix_b,
            equals, result
        ).arrange(RIGHT)
        
        # Create step-by-step calculations
        calc1 = Text("= [2(1) + 1(2)]")
        calc2 = Text("= [2 + 2]")
        calc3 = Text("= [4]")
        
        calcs = VGroup(calc1, calc2, calc3).arrange(DOWN)
        calcs.next_to(equation, DOWN, buff=1)
        
        # Create animations
        self.play(Create(matrix_a))
        self.play(Create(matrix_b))
        self.play(Write(times), Write(equals))
        self.play(Create(result))
        self.wait()
        
        self.play(Write(calc1))
        self.play(Write(calc2))
        self.play(Write(calc3))
        self.wait()'''

def generate_eigenvalue_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create matrix and vector
        matrix = VGroup(
            Text("2  1"),
            Text("1  2")
        ).arrange(DOWN)
        matrix.add(SurroundingRectangle(matrix))
        
        vector = VGroup(
            Text("v₁"),
            Text("v₂")
        ).arrange(DOWN)
        vector.add(SurroundingRectangle(vector))
        
        # Create lambda and equation
        lambda_text = Text("λ")
        equation = Text("Av = λv")
        
        # Position everything
        group = VGroup(matrix, vector, lambda_text, equation).arrange(RIGHT)
        group.to_edge(UP)
        
        # Create characteristic equation steps
        char_eq = Text("det(A - λI) = 0")
        expanded = Text("|2-λ  1|")
        expanded2 = Text("|1  2-λ|")
        solved = Text("(2-λ)² - 1 = 0")
        result = Text("λ = 1, 3")
        
        # Position steps
        steps = VGroup(
            char_eq, expanded, expanded2,
            solved, result
        ).arrange(DOWN)
        steps.next_to(group, DOWN, buff=1)
        
        # Create animations
        self.play(Create(matrix), Create(vector))
        self.play(Write(lambda_text), Write(equation))
        self.wait()
        
        self.play(Write(char_eq))
        self.play(Write(expanded), Write(expanded2))
        self.play(Write(solved))
        self.play(Write(result))
        self.wait()'''

def generate_complex_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Set up plane
        plane = ComplexPlane()
        self.play(Create(plane))
        
        # Create complex number
        z = 3 + 2j
        dot = Dot([3, 2, 0], color=YELLOW)
        
        # Create vector and labels
        vector = Arrow(
            ORIGIN, dot.get_center(),
            buff=0, color=YELLOW
        )
        re_line = DashedLine(
            ORIGIN, [3, 0, 0], color=BLUE
        )
        im_line = DashedLine(
            [3, 0, 0], [3, 2, 0], color=RED
        )
        
        # Add labels
        z_label = Text("z = 3 + 2i", font_size=36)
        z_label.next_to(dot, UR)
        re_label = Text("Re(z) = 3", font_size=36)
        re_label.next_to(re_line, DOWN)
        im_label = Text("Im(z) = 2", font_size=36)
        im_label.next_to(im_line, RIGHT)
        
        # Animations
        self.play(Create(vector))
        self.play(Write(z_label))
        self.wait()
        self.play(
            Create(re_line),
            Create(im_line)
        )
        self.play(
            Write(re_label),
            Write(im_label)
        )
        self.wait()'''

def generate_diff_eq_code():
    return '''from manim import *
import numpy as np

class MainScene(Scene):
    def construct(self):
        # Create differential equation
        eq = MathTex(r"\\frac{dy}{dx} + 2y = e^x")
        
        # Solution steps
        step1 = MathTex(r"y = e^{-2x}\\int e^x \\cdot e^{2x} dx")
        step2 = MathTex(r"y = e^{-2x}\\int e^{3x} dx")
        step3 = MathTex(r"y = e^{-2x} \\cdot \\frac{1}{3}e^{3x} + Ce^{-2x}")
        step4 = MathTex(r"y = \\frac{1}{3}e^x + Ce^{-2x}")
        
        # Arrange equations
        VGroup(
            eq, step1, step2, step3, step4
        ).arrange(DOWN, buff=0.5)
        
        # Create graph
        axes = Axes(
            x_range=[-2, 2],
            y_range=[-2, 2],
            axis_config={"include_tip": True}
        )
        
        # Plot particular solution (C=0)
        graph = axes.plot(
            lambda x: (1/3)*np.exp(x),
            color=YELLOW
        )
        
        # Animations
        self.play(Write(eq))
        self.wait()
        self.play(Write(step1))
        self.wait()
        self.play(Write(step2))
        self.wait()
        self.play(Write(step3))
        self.wait()
        self.play(Write(step4))
        self.wait()
        
        # Show graph
        self.play(
            FadeOut(VGroup(eq, step1, step2, step3, step4))
        )
        self.play(Create(axes), Create(graph))
        self.wait()'''

def generate_trig_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create coordinate plane
        plane = NumberPlane(
            x_range=[-4, 4],
            y_range=[-2, 2],
            axis_config={"include_tip": True}
        )
        
        # Add custom labels
        x_label = Text("x").next_to(plane.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(plane.y_axis.get_end(), UP)
        
        # Create unit circle
        circle = Circle(radius=1, color=BLUE)
        
        # Create angle tracker
        theta = ValueTracker(0)
        
        # Create dot that moves around circle
        dot = always_redraw(
            lambda: Dot(
                circle.point_at_angle(theta.get_value()),
                color=YELLOW
            )
        )
        
        # Create lines to show sine and cosine
        x_line = always_redraw(
            lambda: Line(
                start=[circle.point_at_angle(theta.get_value())[0], 0, 0],
                end=circle.point_at_angle(theta.get_value()),
                color=GREEN
            )
        )
        
        y_line = always_redraw(
            lambda: Line(
                start=[0, 0, 0],
                end=[circle.point_at_angle(theta.get_value())[0], 0, 0],
                color=RED
            )
        )
        
        # Create labels
        sin_label = Text("sin(θ)").next_to(x_line).set_color(GREEN)
        cos_label = Text("cos(θ)").next_to(y_line).set_color(RED)
        
        # Add everything to scene
        self.play(Create(plane), Write(x_label), Write(y_label))
        self.play(Create(circle))
        self.play(Create(dot))
        self.play(Create(x_line), Create(y_line))
        self.play(Write(sin_label), Write(cos_label))
        
        # Animate angle
        self.play(
            theta.animate.set_value(2*PI),
            run_time=4,
            rate_func=linear
        )
        self.wait()'''

def generate_quadratic_code():
    return '''from manim import *

class MainScene(Scene):
    def construct(self):
        # Create coordinate system
        axes = Axes(
            x_range=[-4, 4],
            y_range=[-2, 8],
            axis_config={"include_tip": True}
        )
        
        # Add custom labels
        x_label = Text("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(axes.y_axis.get_end(), UP)
        
        # Create quadratic function
        def func(x):
            return x**2
            
        graph = axes.plot(
            func,
            color=BLUE,
            x_range=[-3, 3]
        )
        
        # Create labels and equation
        equation = Text("f(x) = x²").to_corner(UL)
        
        # Create dot and value tracker
        x = ValueTracker(-3)
        dot = always_redraw(
            lambda: Dot(
                axes.c2p(
                    x.get_value(),
                    func(x.get_value())
                ),
                color=YELLOW
            )
        )
        
        # Create lines to show x and y values
        v_line = always_redraw(
            lambda: axes.get_vertical_line(
                axes.input_to_graph_point(
                    x.get_value(),
                    graph
                ),
                color=RED
            )
        )
        h_line = always_redraw(
            lambda: axes.get_horizontal_line(
                axes.input_to_graph_point(
                    x.get_value(),
                    graph
                ),
                color=GREEN
            )
        )
        
        # Add everything to scene
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(graph))
        self.play(Write(equation))
        self.play(Create(dot), Create(v_line), Create(h_line))
        
        # Animate x value
        self.play(
            x.animate.set_value(3),
            run_time=6,
            rate_func=there_and_back
        )
        self.wait()'''

def generate_3d_surface_code():
    return '''from manim import *
import numpy as np

class MainScene(ThreeDScene):
    def construct(self):
        # Configure the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Create axes
        axes = ThreeDAxes()
        
        # Create surface
        def func(x, y):
            return np.sin(x) * np.cos(y)
            
        surface = Surface(
            lambda u, v: axes.c2p(u, v, func(u, v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=32,
            checkerboard_colors=[BLUE_D, BLUE_E]
        )
        
        # Add custom labels
        x_label = Text("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(axes.y_axis.get_end(), UP)
        z_label = Text("z").next_to(axes.z_axis.get_end(), OUT)
        
        # Create animations
        self.begin_ambient_camera_rotation(rate=0.2)
        self.play(Create(axes), Write(x_label), Write(y_label), Write(z_label))
        self.play(Create(surface))
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.wait()'''

def generate_sphere_code():
    return '''from manim import *
import numpy as np

class MainScene(ThreeDScene):
    def construct(self):
        # Configure the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Create axes
        axes = ThreeDAxes()
        
        # Create sphere
        sphere = Surface(
            lambda u, v: np.array([
                np.cos(u) * np.cos(v),
                np.cos(u) * np.sin(v),
                np.sin(u)
            ]),
            u_range=[-PI/2, PI/2],
            v_range=[0, TAU],
            checkerboard_colors=[BLUE_D, BLUE_E]
        )
        
        # Add custom labels
        x_label = Text("x").next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y").next_to(axes.y_axis.get_end(), UP)
        z_label = Text("z").next_to(axes.z_axis.get_end(), OUT)
        
        # Create animations
        self.begin_ambient_camera_rotation(rate=0.2)
        self.play(Create(axes), Write(x_label), Write(y_label), Write(z_label))
        self.play(Create(sphere))
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.wait()'''

def generate_manim_code(concept):
    """Generate Manim code based on the concept with validation."""
    try:
        # FORCE AI GENERATION: Disable templates to ensure step-by-step videos
        app.logger.info("Using AI generation for Manim code")
        
        # Detect if this is an equation-solving problem
        concept_lower = concept.lower()
        is_equation = any(keyword in concept_lower for keyword in ["solve", "=", "equation", "find x", "find y"])
        
        # Generate code (with retry if validation fails)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                code = generate_ai_manim_code(concept)
            except ValueError as ve:
                # Re-raise ValueError (empty output) immediately without retry
                if "empty output" in str(ve).lower():
                    logger.error(f"LLM returned empty output, not retrying: {ve}")
                    raise
                # Other ValueErrors can be retried
                if attempt < max_retries - 1:
                    logger.warning(f"Code generation failed, retrying (attempt {attempt + 1}/{max_retries}): {ve}")
                    continue
                else:
                    raise
            
            # Explicitly check if extracted code is empty or whitespace
            if not code or not code.strip():
                if attempt < max_retries - 1:
                    logger.warning(f"Empty code generated, retrying (attempt {attempt + 1}/{max_retries})")
                    continue
                else:
                    raise ValueError("LLM returned empty output")
            
            # Basic validation: Check for required components
            code_lower = code.lower()
            required_components = {
                "from manim import": "from manim import" in code_lower,
                "class": "class" in code_lower,
                "scene": "scene" in code_lower,
                "def construct": "def construct" in code_lower
            }
            
            # Map back to display names for error message
            display_names = {
                "from manim import": "from manim import",
                "class": "class",
                "scene": "Scene",
                "def construct": "def construct"
            }
            
            missing_components = [display_names[comp] for comp, present in required_components.items() if not present]
            
            if missing_components:
                app.logger.error(f"VALIDATION FAILED: Missing required components: {missing_components}")
                app.logger.error(f"Rejected code (first 500 chars):\n{code[:500]}")
                if attempt < max_retries - 1:
                    app.logger.warning(f"Retrying generation (attempt {attempt + 1}/{max_retries})")
                    continue
                else:
                    raise ValueError(
                        f"Generated code is invalid: missing required components: {missing_components}. "
                        f"Code must contain: from manim import, class, Scene, def construct"
                    )
            
            # Hard-fail validation: Reject graph generation for equations
            if is_equation:
                forbidden_patterns = ["Axes", "NumberPlane", "plot(", "GraphScene"]
                code_upper = code.upper()
                
                found_forbidden = [pattern for pattern in forbidden_patterns if pattern.upper() in code_upper]
                
                if found_forbidden:
                    app.logger.error(f"VALIDATION FAILED: Found forbidden patterns in generated code: {found_forbidden}")
                    app.logger.error(f"Rejected code (first 500 chars):\n{code[:500]}")
                    if attempt < max_retries - 1:
                        app.logger.warning(f"Retrying generation (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        raise ValueError(
                            f"Generated code contains forbidden graph patterns for equation solving: {found_forbidden}. "
                            f"Code must use MathTex and TransformMatchingTex only."
                        )
            
            # Code passed validation
            app.logger.info("Manim code generated and validated successfully")
            return code
            
    except Exception as e:
        app.logger.error(f"Error generating Manim code: {str(e)}")
        raise

def generate_basic_visualization_code():
    """Generate code for basic visualization."""
    return '''from manim import *
import numpy as np

class MainScene(Scene):
    def construct(self):
        # Create title
        title = Text("Mathematical Visualization", font_size=36).to_edge(UP)
        
        # Create axes
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-3, 3, 1],
            axis_config={"include_tip": True},
            x_length=10,
            y_length=6
        )
        
        # Add labels
        x_label = Text("x", font_size=24).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = Text("y", font_size=24).next_to(axes.y_axis.get_end(), UP)
        
        # Create function graphs
        sin_graph = axes.plot(lambda x: np.sin(x), color=BLUE)
        cos_graph = axes.plot(lambda x: np.cos(x), color=RED)
        
        # Create labels for functions
        sin_label = Text("sin(x)", font_size=24, color=BLUE).next_to(sin_graph, UP)
        cos_label = Text("cos(x)", font_size=24, color=RED).next_to(cos_graph, DOWN)
        
        # Create dot to track movement
        moving_dot = Dot(color=YELLOW)
        moving_dot.move_to(axes.c2p(-5, 0))
        
        # Create path for dot to follow
        path = VMobject()
        path.set_points_smoothly([
            axes.c2p(x, np.sin(x)) 
            for x in np.linspace(-5, 5, 100)
        ])
        
        # Animate everything
        self.play(Write(title))
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(sin_graph), Write(sin_label))
        self.play(Create(cos_graph), Write(cos_label))
        self.play(Create(moving_dot))
        
        # Animate dot following the sine curve
        self.play(
            MoveAlongPath(moving_dot, path),
            run_time=3,
            rate_func=linear
        )
        
        # Final pause
        self.wait()
'''

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        concept = request.json.get('concept', '')
        if not concept:
            return jsonify({'error': 'No concept provided'}), 400
            
        concept = sanitize_input(concept)
        
        # Determine render quality
        quality_requested = request.json.get('quality', RENDER_QUALITY_DEFAULT).lower()
        if quality_requested not in {'low', 'medium', 'high'}:
            quality_requested = RENDER_QUALITY_DEFAULT
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
        filename = f'scene_{timestamp}_{random_str}'
        
        # Create temporary directory for this generation
        temp_dir = os.path.join(app.config['TEMP_DIR'], filename)
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Check if this is a LaTeX expression
            used_ai = False
            
            # Get appropriate template or generate code
            if is_likely_latex(concept):
                manim_code = generate_latex_scene_code(concept)
            else:
                # Use centralized generation function (which forces AI usage)
                try:
                    manim_code = generate_manim_code(concept)
                    # Check if it was AI generated (generate_manim_code returns None or a string)
                    # Note: generate_manim_code calls generate_ai_manim_code which is AI.
                    used_ai = True
                except ValueError as ve:
                    # Validation error from generate_manim_code
                    logger.error(f'Manim code validation failed: {ve}')
                    return jsonify({
                        'error': 'Failed to generate valid Manim code',
                        'details': str(ve)
                    }), 500
                except Exception as gen_err:
                    logger.error(f'Error generating Manim code: {gen_err}', exc_info=True)
                    return jsonify({
                        'error': 'Failed to generate Manim code',
                        'details': str(gen_err)
                    }), 500
            
            if not manim_code:
                return jsonify({'error': 'Failed to generate code template'}), 500
            
            # Validate that the generated code contains a Scene class (preferably MainScene)
            # Check for MainScene first, then fall back to any Scene class
            if 'class MainScene' not in manim_code:
                # Try to find any Scene class
                if 'class' in manim_code and 'Scene' in manim_code:
                    # Extract class name and update the code to use MainScene
                    import re
                    class_match = re.search(r'class\s+(\w+)\s*\(.*Scene', manim_code)
                    if class_match:
                        actual_class = class_match.group(1)
                        logger.warning(f'Generated code uses class {actual_class}, replacing with MainScene')
                        # Replace the class name with MainScene
                        manim_code = re.sub(rf'class\s+{actual_class}\s*\(', 'class MainScene(', manim_code)
                        # Also update the scene name in the command if needed
                    else:
                        logger.error('Generated code does not contain a valid Scene class')
                        logger.debug(f'Generated code: {manim_code[:500]}...')
                        return jsonify({
                            'error': 'Generated code is invalid: Scene class not found',
                            'details': 'The AI generated code does not match the expected structure.'
                        }), 500
                else:
                    logger.error('Generated code does not contain a Scene class')
                    logger.debug(f'Generated code: {manim_code[:500]}...')
                    return jsonify({
                        'error': 'Generated code is invalid: Scene class not found',
                        'details': 'The AI generated code does not match the expected structure.'
                    }), 500
            
            # Write code to temporary file
            code_file = os.path.join(temp_dir, 'scene.py')
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(manim_code)
            
            # Create media directory
            media_dir = os.path.join(temp_dir, 'media')
            os.makedirs(media_dir, exist_ok=True)
            
            # Determine manim quality flag
            quality_flag = {'low': '-ql', 'medium': '-qm', 'high': '-qh'}[quality_requested]
            
            # Run manim command with error handling
            output_file = os.path.join(app.static_folder, 'videos', f'{filename}.mp4')
            command = [
                sys.executable,  # Use current Python interpreter
                '-m', 'manim',
                'render',
                quality_flag,
                '--format', 'mp4',
                '--media_dir', media_dir,
                code_file,
                'MainScene'
            ]
            
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    cwd=temp_dir,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode != 0:
                    # Capture both stderr and stdout for better error reporting
                    error_msg = result.stderr if result.stderr else result.stdout if result.stdout else 'Unknown error during animation generation'
                    logger.error(f'Manim execution failed (returncode={result.returncode})')
                    logger.error(f'Manim stderr: {result.stderr}')
                    logger.error(f'Manim stdout: {result.stdout}')
                    # Raise RuntimeError as requested, but we'll catch it in the outer handler
                    raise RuntimeError(f'Manim render failed: {error_msg}')
                
                # Look for the video file in multiple possible locations
                possible_paths = [
                    os.path.join(media_dir, 'videos', 'scene', '1080p60', 'MainScene.mp4'),
                    os.path.join(media_dir, 'videos', 'scene', '720p30', 'MainScene.mp4'),
                    os.path.join(media_dir, 'videos', 'scene', '480p15', 'MainScene.mp4'),  # low-quality default in manim 0.17
                    os.path.join(media_dir, 'videos', 'MainScene.mp4'),
                    os.path.join(temp_dir, 'MainScene.mp4')
                ]
                
                video_found = False
                for source_path in possible_paths:
                    if os.path.exists(source_path):
                        shutil.move(source_path, output_file)
                        video_found = True
                        break
                
                # Fallback: walk media_dir recursively to locate the file
                if not video_found:
                    for root, _dirs, files in os.walk(media_dir):
                        if 'MainScene.mp4' in files:
                            try:
                                shutil.move(os.path.join(root, 'MainScene.mp4'), output_file)
                                video_found = True
                                break
                            except Exception as move_err:
                                logger.error(f'Error moving located video: {move_err}')
                                # if move fails, continue searching
                                continue
                
                if not video_found:
                    logger.error(f'Video not found in any of these locations or recursively under media_dir: {possible_paths}')
                    return jsonify({'error': 'Generated video file not found'}), 500
                
                # Generate explanation
                explanation = generate_explanation(concept)

                # Return success response
                return jsonify({
                    'success': True,
                    'video_url': url_for('static', filename=f'videos/{filename}.mp4'),
                    'code': manim_code,
                    'used_ai': used_ai,
                    'render_quality': quality_requested,
                    'explanation': explanation
                })
                
            except subprocess.TimeoutExpired:
                return jsonify({
                    'error': 'Animation generation timed out',
                    'details': 'The animation took too long to generate. Please try a simpler concept.'
                }), 500
            except RuntimeError as re:
                # Manim execution failed
                logger.error(f'Manim execution RuntimeError: {re}')
                return jsonify({
                    'error': 'Failed to generate animation',
                    'details': str(re)
                }), 500
                
        finally:
            # Cleanup temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.error(f'Error generating animation: {str(e)}', exc_info=True)
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f'Full traceback: {error_trace}')
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/static/videos/<path:filename>')
def serve_video(filename):
    """Serve video files from static/videos directory."""
    try:
        return send_from_directory(
            os.path.join(app.root_path, 'static', 'videos'),
            filename,
            mimetype='video/mp4'
        )
    except Exception as e:
        app.logger.error(f"Error serving video {filename}: {str(e)}")
        return jsonify({'error': 'Video not found'}), 404

@app.route('/demos', methods=['GET'])
def get_demos():
    """Return the 4 specific demo GIFs for the landing page."""
    # Define the 4 specific demos in order with exact GIF filenames
    demos = [
        {
            'filename': 'differential_equations.gif',
            'title': 'Trigonometry',
            'description': 'Visualization of sine and cosine functions on the unit circle with animated angle.'
        },
        {
            'filename': '3d_calculus.gif',
            'title': '3D Surface Plot',
            'description': '3D visualization of the surface area of a cube with animations.'
        },
        {
            'filename': 'ComplexNumbersAnimation_ManimCE_v0.17.3.gif',
            'title': 'Complex Numbers',
            'description': 'Geometric interpretation of complex number operations with rotation and scaling.'
        },
        {
            'filename': 'TrigonometryAnimation_ManimCE_v0.17.3.gif',
            'title': 'Linear Algebra',
            'description': 'Differential equations to life by visualizing solution curves and phase spaces.'
        }
    ]
    
    videos = []
    for demo in demos:
        filepath = os.path.join(app.static_folder, 'gifs', demo['filename'])
        if os.path.exists(filepath):
            videos.append({
                'filename': demo['filename'],
                'title': demo['title'],
                'description': demo['description'],
                'url': url_for('static', filename=f'gifs/{demo["filename"]}')
            })
    
    return jsonify({'videos': videos})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
