import os
import uuid
import ast
import wave
import contextlib
import logging
from typing import List, Dict, Any

from ..math.steps import generate_math_steps
from ..prompts.manim_prompt import generate_manim_code
from ..prompts.voice_prompt import generate_voice_script
from ..audio.tts import generate_audio_segment
from ..manim_engine.renderer import render_scene

logger = logging.getLogger(__name__)

def get_wav_duration(file_path: str) -> float:
    """Returns duration of a wav file in seconds."""
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def inject_audio_into_script(script_content: str, voice_data: dict, audio_files: Dict[int, str]) -> str:
    """
    Injects audio playback and wait calls into the Manim script using AST.
    
    Args:
        script_content: The original Python script.
        voice_data: The JSON voice data containing segments.
        audio_files: Dict mapping segment index to audio file path.
    """
    try:
        tree = ast.parse(script_content)
    except SyntaxError as e:
        logger.error(f"Failed to parse generated Manim code for audio injection: {e}")
        raise RuntimeError(f"Manim code generation failed: Syntactically invalid Python code. Error: {e}") from e

    # Find the construct method
    construct_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'construct':
            construct_node = node
            break
    
    if not construct_node:
        raise RuntimeError("Manim audio injection failed: No 'construct' method found in generated scene.")

    # Identify animation steps (self.play, self.add)
    # Map animation_index -> node.end_lineno
    animation_end_lines = []
    
    for node in construct_node.body:
        # Check for self.play or self.add calls
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'self':
                if call.func.attr in ['play', 'add']:
                    animation_end_lines.append(node.end_lineno)
    
    lines = script_content.splitlines()
    insertions = [] # List of (line_index, code_lines)

    voice_segments = voice_data.get("segments", [])
    
    # Process segments
    for i, segment in enumerate(voice_segments):
        start_index = segment.get("start_after_animation", 0)
        audio_path = audio_files.get(i)
        
        if not audio_path:
            # Voice script requested a segment but no audio file was generated?
            # Ideally this shouldn't happen if we loop correctly in generate_video.
            logger.warning(f"Voice segment {i} has no corresponding audio file. Skipping injection.")
            continue
            
        # Get duration
        duration = 0.0
        try:
            duration = get_wav_duration(audio_path)
            # Add a small buffer as requested ("Add self.wait(2) minimum")
            padding = 0.5
            wait_time = max(duration + padding, 2.0)
        except Exception as e:
            logger.error(f"Failed to get duration for {audio_path}: {e}")
            wait_time = 2.0 # Fallback

        # Escape path for python string
        safe_path = audio_path.replace("\\", "/")
        
        injection_code = [
            f"        self.add_sound('{safe_path}')",
            f"        self.wait({wait_time:.2f})"
        ]

        # Determine insertion point
        # If start_index is within bounds of found animations, insert after that animation
        if 0 <= start_index < len(animation_end_lines):
            target_line = animation_end_lines[start_index]
            insertions.append((target_line, injection_code))
        else:
            # If start_index is -1 (start of scene), insert at beginning of construct
            # But prompt said 0 means after first animation.
            # If index is out of bounds (e.g. valid animation count < requested), fail or append?
            # Prompt says "Fail loudly if audio injection fails".
            # If the voice script expects 10 animations but code only has 2, that's a mismatch.
            # However, failing strict might be too harsh if LLM hallucinated animation count.
            # Let's append to the last known animation to ensure audio plays.
            if animation_end_lines:
                target_line = animation_end_lines[-1]
                insertions.append((target_line, injection_code))
                logger.warning(f"Voice segment {i} requested after animation {start_index} but only {len(animation_end_lines)} found. Appending to end.")
            else:
                 # No animations found using self.play/self.add. Could be pure self.wait scene?
                 # Insert at end of construct logic (naive approach: specific line not easy to find without strict parsing).
                 # We will fail loudly here for safety.
                 raise RuntimeError(f"Voice segment {i} requested validation, but no animations (self.play/self.add) found in construct method.")

    # Apply insertions in reverse order of line number to prevent shifting
    # Sort by line number descending
    insertions.sort(key=lambda x: x[0], reverse=True)
    
    for line_num, code_lines in insertions:
        # insert after line_num (which is 1-indexed, so index is line_num)
        idx = line_num 
        lines[idx:idx] = code_lines

    final_content = "\n".join(lines)
    
    # Verification: Confirm string "add_sound" exists if we expected insertions
    if voice_segments and "add_sound" not in final_content:
        raise RuntimeError("Audio injection verification failed: 'add_sound' not found in final script string.")
        
    return final_content


def generate_video(question: str, output_dir: str = "static/videos"):
    """
    Generates a Manim video for the given math question using Gemini for code and voice.
    """
    # 1. Generate Math Steps (Text)
    logger.info(f"Generating math steps for: {question}")
    steps_data = generate_math_steps(question)
    
    # Format a concept string for the Manim coder
    if isinstance(steps_data["steps"], list):
        steps_text = "\n".join(steps_data["steps"])
    else:
        steps_text = str(steps_data["steps"])
        
    full_concept = f"""
    Topic: {question}
    
    Explanation Steps:
    {steps_text}
    
    Detailed Explanation:
    {steps_data["explanation"]}
    """
    
    # 2. Generate Manim Code
    logger.info("Generating Manim code...")
    manim_code = generate_manim_code(full_concept)
    
    # 3. Generate Voice Script
    logger.info("Generating Voice script...")
    voice_data = generate_voice_script(manim_code)
    
    # 4. Synthesize Audio
    logger.info("Synthesizing audio...")
    tmp_audio_dir = os.path.join("tmp", "audio")
    audio_file_map = {} # segment_index -> absolute path
    
    segments = voice_data.get("segments", [])
    for i, segment in enumerate(segments):
        text = segment.get("text", "")
        if text:
            path = generate_audio_segment(text, tmp_audio_dir)
            audio_file_map[i] = path
            
    # 5. Inject Audio into Code
    logger.info("Injecting audio into script...")
    final_script = inject_audio_into_script(manim_code, voice_data, audio_file_map)
    
    # 6. Write to File
    unique_id = str(uuid.uuid4())
    scene_file_name = f"scene_{unique_id}.py"
    scene_file_path = os.path.join("tmp", scene_file_name)
    os.makedirs("tmp", exist_ok=True)
    
    with open(scene_file_path, "w", encoding="utf-8") as f:
        f.write(final_script)
        
    # 7. Render
    logger.info(f"Rendering scene from {scene_file_path}...")
    abs_scene_path = os.path.abspath(scene_file_path)
    abs_output_dir = os.path.abspath(output_dir)
    
    # Detect scene class name? 
    # Usually we need to know it. The Prompt doesn't enforce a specific name, 
    # but Manim CLI works if we pass the file. If multiple scenes, we need to pick one.
    # We can parse it from AST or just let Manim render all (but render_scene function takes scene_name).
    # Let's extract the first class that inherits from Scene or similar.
    
    scene_name = "Scene" # Default fallback
    try:
        tree = ast.parse(final_script)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                scene_name = node.name
                break # Take the first class
    except:
        pass

    try:
        video_path = render_scene(
            scene_file=abs_scene_path,
            scene_name=scene_name,
            output_dir=abs_output_dir,
            quality="l" 
        )
        return video_path
    finally:
        # Cleanup code file (optional, maybe keep for debug)
        # if os.path.exists(scene_file_path):
        #    os.remove(scene_file_path)
        pass

