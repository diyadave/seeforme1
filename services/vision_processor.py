#!/usr/bin/env python3
"""
Online Vision Processor for SeeForMe (Web-Optimized)
- Uses Gemini Pro Vision for unified scene and emotion detection.
- Methods accept image frames, making them callable from web endpoints.
- Compresses images before sending to the API for speed.
- Gracefully handles missing or invalid API keys.
"""

import os
import cv2
import base64
import logging
import google.generativeai as genai  
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from io import BytesIO
from PIL import Image
import re

load_dotenv()
logger = logging.getLogger(__name__)

class OnlineVisionProcessor:
    def __init__(self):
        self.vision_model = None
        self.api_key_valid = False
        self._initialize_vision()
        logger.info("👁️ Online Vision Processor (Web-Optimized) initialized")
        
    def _initialize_vision(self):
        """Initialize Gemini Vision model with robust error handling."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ GEMINI_API_KEY not found in .env. Vision services will be disabled.")
            return

        try:
            genai.configure(api_key=api_key)
           
            print(">>> SDK version:", genai.__version__)
            print(">>> Available models:")
            for m in genai.list_models():
                if "gemini" in m.name:
                    print("   ", m.name)
            self.vision_model = genai.GenerativeModel("models/gemini-2.5-flash")


            # Quick call to check validity
            genai.list_models()
            self.api_key_valid = True
            logger.info("✅ Gemini Vision model loaded. API key is valid.")
        except Exception as e:
            logger.error(f"❌ Gemini Vision initialization failed. The API key may be invalid or expired: {e}")
            self.vision_model = None

    def _process_frame_for_gemini(self, frame: bytes, resize_dim=(640, 480)) -> Optional[Image.Image]:
        """Converts raw image bytes to a PIL Image object for Gemini."""
        if not frame:
            return None
        try:
            image = Image.open(BytesIO(frame)).convert("RGB")
            image.thumbnail(resize_dim)  # resize while maintaining aspect ratio
            return image
        except Exception as e:
            logger.error(f"❌ Failed to process image frame: {e}")
            return None

    def analyze_scene(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyzes a scene from image bytes using Gemini Vision."""
        if not self.api_key_valid or not self.vision_model:
            return self._get_error_response("Vision service is not configured. Check API key.")

        pil_image = self._process_frame_for_gemini(image_bytes)
        if not pil_image:
            return self._get_error_response("Invalid image data received.")

        prompt = """You are an assistant for a visually impaired user. Describe the scene in this image clearly and concisely.
        Focus on:
        1. The overall environment (e.g., 'You are in a kitchen.').
        2. Key objects and their relative positions (e.g., 'There is a red mug on the table to your left.').
        3. Any people and their actions.
        4. Potential obstacles or hazards.
        Keep the description under 75 words."""
        try:
            response = self.vision_model.generate_content([prompt, pil_image])
            description = getattr(response, "text", "").strip() or "Scene description unavailable."
            return {'status': 'ok', 'description': description, 'analysis_type': 'scene'}
        except Exception as e:
            logger.error(f"❌ Gemini scene analysis failed: {e}")
            return self._get_error_response("Scene analysis is currently unavailable.")

    def detect_emotion(self, image_bytes: bytes) -> Dict[str, Any]:
        """Detects emotion from a face in image bytes using Gemini Vision."""
        if not self.api_key_valid or not self.vision_model:
            return self._get_error_response("Vision service is not configured.", 'emotion')

        pil_image = self._process_frame_for_gemini(image_bytes)
        if not pil_image:
            return self._get_error_response("Invalid image data for emotion detection.", 'emotion')

        prompt = """Analyze the primary facial expression in this image.
        Respond with only ONE word from this list: [happy, sad, angry, surprise, neutral].
        If no face is visible or the expression is unclear, respond with 'neutral'."""
        try:
            response = self.vision_model.generate_content([prompt, pil_image])
            emotion = getattr(response, "text", "neutral").strip().lower()
            valid_emotions = {'happy', 'sad', 'angry', 'surprise', 'neutral'}
            if emotion not in valid_emotions:
                emotion = 'neutral'
            
            return {
                'status': 'ok',
                'emotion': emotion,
                'confidence': 0.9 if emotion != 'neutral' else 0.6,
                'analysis_type': 'emotion'
            }
        except Exception as e:
            logger.error(f"❌ Gemini emotion detection failed: {e}")
            return self._get_error_response("Emotion detection is currently unavailable.", 'emotion')

    def _get_error_response(self, message: str, analysis_type: str = 'scene') -> Dict[str, Any]:
        """Returns a standardized error response dictionary."""
        response = {'status': 'error', 'description': message, 'analysis_type': analysis_type}
        if analysis_type == 'emotion':
            response.update({'emotion': 'error', 'confidence': 0.0})
        return response


# Global instance
vision_processor = OnlineVisionProcessor()
