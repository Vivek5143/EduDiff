import google.generativeai as genai
import sys

try:
    print(f"genai version: {genai.__version__}")
except:
    print("genai version: unknown")

try:
    client = genai.Client
    print("genai.Client exists")
except AttributeError:
    print("genai.Client DOES NOT exist")
