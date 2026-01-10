
import os
import sys

# Add the current directory to sys.path so we can import edudiff
sys.path.append(os.getcwd())

from dotenv import load_dotenv
load_dotenv()
from edudiff.pipeline.generate import generate_video

def main():
    print(f"CWD: {os.getcwd()}")
    key = os.getenv("GEMINI_API_KEY")
    print(f"Key present: {bool(key)}")
    if not key:
        print("Available env vars:", [k for k in os.environ.keys() if "API" in k])
    print("Starting pipeline test...")
    question = "Explain the concept of a right-angled triangle and key terms."
    
    try:
        video_path = generate_video(question)
        print(f"SUCCESS: Video generated at: {video_path}")
    except Exception as e:
        print(f"FAILURE: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
