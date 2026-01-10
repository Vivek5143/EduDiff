import os
import sys
import logging

# Configure logging to show info messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add backend directory to path so we can import edudiff
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from edudiff.pipeline.generate import generate_video
except Exception as e:
    logging.error(f"Failed to import edudiff: {e}")
    sys.exit(1)

def test_generate():
    logging.info("Testing video generation...")
    question = "Expand (x + 1)^2 step by step"
    logging.info(f"Question: {question}")
    try:
        video_path = generate_video(question)
        if video_path and os.path.exists(video_path):
            logging.info(f"SUCCESS: Video generated at {video_path}")
        else:
            logging.error(f"FAILURE: Video path returned but file not found or None returned. Path: {video_path}")
    except Exception as e:
        logging.error(f"FAILURE: Exception occurred: {e}", exc_info=True)

if __name__ == "__main__":
    test_generate()
