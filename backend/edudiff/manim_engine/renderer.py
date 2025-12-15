import subprocess
import os
import sys
import logging

logger = logging.getLogger(__name__)

def render_scene(scene_file, scene_name, output_dir, quality="l"):
    """
    Renders a specific Manim scene.
    
    Args:
        scene_file (str): Path to the python file containing the scene.
        scene_name (str): Name of the scene class.
        output_dir (str): Directory to save the output video.
        quality (str): Quality flag ('l', 'm', 'h', 'p', 'k'). Default 'l' (low) for speed.
    
    Returns:
        str: Path to the generated video file.
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct command
    # manim -q[quality] --media_dir [output_dir] [scene_file] [scene_name]
    # Note: Manim's output structure is fixed relative to media_dir. 
    # We might need to move the file after rendering to flatten the structure if desired.
    
    command = [
        sys.executable, "-m", "manim",
        "-q" + quality,
        "--media_dir", output_dir,
        scene_file,
        scene_name
    ]
    
    logger.info(f"Running command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=300 # 5 minute timeout
        )
        logger.info("Render successful")
        
        # Manim default output path structure:
        # media_dir/videos/scene_file_name/quality/scene_name.mp4
        # We need to find this file.
        
        scene_file_name = os.path.splitext(os.path.basename(scene_file))[0]
        quality_map = {'l': '480p15', 'm': '720p30', 'h': '1080p60', 'p': '2160p60', 'k': '4320p60'}
        # Note: Manim might change these folder names based on exact version and framerate settings.
        # But for standard settings:
        
        # Let's search for the .mp4 file in the output_dir
        video_path = None
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".mp4") and scene_name in file:
                    video_path = os.path.join(root, file)
                    break
            if video_path:
                break
        
        if video_path:
            return video_path
        else:
            logger.error("Video file not found after rendering")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Render failed: {e.stderr}")
        raise RuntimeError(f"Manim render failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("Render timed out")
        raise RuntimeError("Manim render timed out")
    except FileNotFoundError:
        logger.error("Manim command not found. Ensure Manim is installed and in PATH.")
        raise RuntimeError("Manim command not found. Please ensure 'manim' is installed and available in your environment.")

