"""
Generate 4 demo videos for EduDiff landing page
Run this script to create the 4 specific demo videos
"""
import os
import sys
import subprocess

# Add the backend directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app import (
    generate_trig_code,
    generate_3d_surface_code,
    generate_complex_code,
    generate_matrix_code
)

DEMOS = [
    {
        'name': 'Trigonometry',
        'filename': 'demo_trigonometry.mp4',
        'code_generator': generate_trig_code
    },
    {
        'name': '3D Surface Plot',
        'filename': 'demo_3d_surface.mp4',
        'code_generator': generate_3d_surface_code
    },
    {
        'name': 'Complex Numbers',
        'filename': 'demo_complex_numbers.mp4',
        'code_generator': generate_complex_code
    },
    {
        'name': 'Linear Algebra',
        'filename': 'demo_linear_algebra.mp4',
        'code_generator': generate_matrix_code
    }
]

def generate_demo_video(name, filename, code_generator):
    print(f"\nGenerating {name}...")
    
    # Generate code
    code = code_generator()
    
    # Create temp file
    temp_file = os.path.join('tmp', f'{filename}.py')
    os.makedirs('tmp', exist_ok=True)
    
    with open(temp_file, 'w') as f:
        f.write(code)
    
    # Render video
    output_file = os.path.join('static', 'videos', filename)
    cmd = [
        'manim',
        'render',
        '-ql',
        '--format', 'mp4',
        temp_file,
        'MainScene',
        '-o', filename
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Find and move the generated video
            media_path = os.path.join('media', 'videos')
            for root, dirs, files in os.walk(media_path):
                if filename in files:
                    src = os.path.join(root, filename)
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    import shutil
                    shutil.move(src, output_file)
                    print(f"✓ {name} generated successfully")
                    return True
        print(f"✗ Failed to generate {name}")
        print(result.stderr)
        return False
    except Exception as e:
        print(f"✗ Error generating {name}: {e}")
        return False

if __name__ == '__main__':
    print("Generating 4 demo videos for EduDiff...")
    print("This may take a few minutes...\n")
    
    for demo in DEMOS:
        generate_demo_video(demo['name'], demo['filename'], demo['code_generator'])
    
    print("\nDone!")
