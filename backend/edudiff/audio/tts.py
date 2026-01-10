import os
import pyttsx3
import uuid

def generate_audio_segment(text: str, output_dir: str) -> str:
    """
    Generates a WAV file for the given text using pyttsx3.
    Returns the absolute path to the generated file.
    """
    engine = pyttsx3.init()
    
    # Configure voice
    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)
    
    # Ensure raw output dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    unique_id = str(uuid.uuid4())
    filename = f"voice_{unique_id}.wav"
    output_path = os.path.join(output_dir, filename)
    abs_path = os.path.abspath(output_path)
    
    # Save to file
    engine.save_to_file(text, abs_path)
    engine.runAndWait()
    
    if not os.path.exists(abs_path):
        raise RuntimeError(f"TTS failed: Expected output file at {abs_path} was not found.")
        
    return abs_path
