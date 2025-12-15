import os
import uuid

from ..math.steps import generate_math_steps
from ..prompts.manim_prompt import generate_manim_code
from ..manim_engine.renderer import render_scene

def generate_video(question: str, output_dir: str = "static/videos"):
    """
    Takes a math question and returns a video filename.
    """
    # 1. Convert question -> math steps
    steps_data = generate_math_steps(question)
    
    # 2. Prepare data for Manim (in this deterministic version, just pass through)
    manim_data = generate_manim_code(steps_data)
    
    # 3. Create a temporary python file to run the scene
    # We need to generate a script that imports EquationTransformScene and instantiates it with our data.
    # Since Manim CLI runs a file, we need to write a file.
    
    unique_id = str(uuid.uuid4())
    scene_file_name = f"scene_{unique_id}.py"
    scene_file_path = os.path.join("tmp", scene_file_name)
    
    # Ensure tmp exists
    os.makedirs("tmp", exist_ok=True)
    
    # Escape strings for python code
    steps_list = str(manim_data["steps"])
    explanation_str = f'"{manim_data["explanation"]}"'
    
    script_content = f"""
from edudiff.manim_engine.equation_transform import EquationTransformScene

class GeneratedScene(EquationTransformScene):
    def __init__(self, **kwargs):
        super().__init__(
            steps={steps_list},
            explanation={explanation_str},
            **kwargs
        )
"""
    
    with open(scene_file_path, "w") as f:
        f.write(script_content)
        
    # 4. Render video
    # We need to pass the absolute path to the scene file because we are running from backend root
    abs_scene_path = os.path.abspath(scene_file_path)
    abs_output_dir = os.path.abspath(output_dir)
    
    try:
        video_path = render_scene(
            scene_file=abs_scene_path,
            scene_name="GeneratedScene",
            output_dir=abs_output_dir,
            quality="l" # Low quality for speed during dev
        )
        return video_path
    finally:
        # Cleanup
        if os.path.exists(scene_file_path):
            os.remove(scene_file_path)
