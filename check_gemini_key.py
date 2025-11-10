#!/usr/bin/env python3
"""
Gemini Vision API Diagnostic Tool
✅ Verifies your GEMINI_API_KEY supports vision-based requests.
✅ Uses a sample image to test scene and emotion description.
"""

import google.generativeai as genai
from dotenv import load_dotenv
import os
from PIL import Image
from io import BytesIO
import requests

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in environment or .env file.")
    exit()

try:
    genai.configure(api_key=api_key)
    print("✅ Gemini API key loaded successfully.\n")
except Exception as e:
    print(f"❌ Failed to configure Gemini API: {e}")
    exit()

# --- Model selection ---
model_name = "models/gemini-2.5-flash"
print(f"🧠 Testing model: {model_name}\n")

try:
    model = genai.GenerativeModel(model_name)
except Exception as e:
    print(f"❌ Failed to load model {model_name}: {e}")
    exit()

# --- Load a small sample image ---
image_path = "sample.jpg"
from PIL import Image
image = Image.open(image_path).convert("RGB")


# --- Test Scene Description ---
try:
    prompt = "Describe this image briefly in one sentence."
    print("\n🔍 Running scene description test...")
    response = model.generate_content([prompt, image])
    print("💬 Scene Description:", response.text.strip())
except Exception as e:
    print(f"❌ Scene description failed: {e}")

# --- Test Emotion Detection ---
try:
    prompt = "Identify the dominant emotion or mood in this image using one word (e.g., happy, sad, neutral)."
    print("\n😊 Running emotion detection test...")
    response = model.generate_content([prompt, image])
    print("💬 Detected Emotion:", response.text.strip())
except Exception as e:
    print(f"❌ Emotion detection failed: {e}")
