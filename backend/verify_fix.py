
import os
import sys
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env variables (simulate app.py behavior)
load_dotenv()

def test_genai_init():
    print(f"GOOGLE_API_KEY from env: {'Set' if os.getenv('GOOGLE_API_KEY') else 'Not Set'}")
    print(f"GEMINI_API_KEY from env: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not Set'}")
    
    GENAI_MODEL = os.getenv('GENAI_MODEL', 'gemini-2.5-flash')
    print(f"Target Model: {GENAI_MODEL}")
    
    # Simulate the init logic
    genai_model = None
    try:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            genai_model = genai.GenerativeModel(GENAI_MODEL)
            print("SUCCESS: GenAI initialized")
            
            # Test generation
            print("Testing content generation...")
            try:
                response = genai_model.generate_content("Say 'Hello World'")
                print(f"Generation Response: {response.text}")
            except Exception as e:
                print(f"Generation FAILED: {e}")
        else:
            print("FAILURE: No API key found")
    except Exception as e:
        print(f"FAILURE: exceptions during init: {e}")

if __name__ == "__main__":
    test_genai_init()
