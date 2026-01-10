
import os
import sys
import logging

# Set up logging before importing app to capture init logs
logging.basicConfig(level=logging.INFO)

# Ensure backend dir is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import generate_ai_manim_code, genai_model, generate_manim_prompt
except ImportError as e:
    logging.error(f"Failed to import from app: {e}")
    sys.exit(1)

def test_app_generation():
    print("Testing generate_ai_manim_code from app.py...")
    
    # Check if model initialized
    if genai_model is None:
        print("FAILURE: genai_model is None in app.py")
        return

    concept = "Expand (x + 1)^2 step by step"
    print(f"Concept: {concept}")
    
    try:
        # We can also check the prompt being generated
        prompt = generate_manim_prompt(concept)
        print(f"Generated Prompt length: {len(prompt)}")
        
        # Call the function
        code = generate_ai_manim_code(concept)
        
        if code and len(code) > 0:
            print("SUCCESS: Code generated successfully")
            print(f"Code snippet: {code[:100]}...")
        else:
            print("FAILURE: Code generation returned empty string")
            
    except Exception as e:
        print(f"FAILURE: Exception during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_app_generation()
