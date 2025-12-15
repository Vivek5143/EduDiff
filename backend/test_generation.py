import os
import sys

# Add backend directory to path so we can import edudiff
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from edudiff.pipeline.generate import generate_video

def test_generate():
    print("Testing video generation...")
    question = "Expand (x + 1)^2"
    print(question)
    try:
        video_path = generate_video(question)
        if video_path and os.path.exists(video_path):
            print(f"SUCCESS: Video generated at {video_path}")
        else:
            print("FAILURE: Video path returned but file not found or None returned")
    except Exception as e:
        print(f"FAILURE: Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generate()
