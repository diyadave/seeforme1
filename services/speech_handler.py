#!/usr/bin/env python3
"""
Online Speech Handler for SeeForMe
- Uses the free Groq API for fast Whisper transcription.
- Supports English, Hindi, and Gujarati as requested.
- Designed to be called by a server endpoint that receives audio from a web client.
"""

import os
import logging
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class OnlineSpeechRecognizer:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            logger.warning("⚠️ GROQ_API_KEY not found in .env file. Speech recognition will fail.")
        self.base_url = "https://api.groq.com/openai/v1/audio/transcriptions"
        logger.info("🎤 Online Speech Recognizer (Groq Whisper API) initialized")

    def transcribe_audio_file(self, audio_file_path: str, language: str = 'en') -> Optional[str]:
        """
        Transcribes an audio file using the Groq Whisper API.

        Args:
            audio_file_path (str): The path to the audio file (e.g., .mp3, .wav).
            language (str): The two-letter language code (e.g., 'en', 'hi', 'gu').

        Returns:
            Optional[str]: The transcribed text, or None if an error occurred.
        """
        if not self.groq_api_key:
            logger.error("Cannot transcribe audio without a Groq API key.")
            return None
        
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found at path: {audio_file_path}")
            return None

        try:
            with open(audio_file_path, 'rb') as audio_file:
                ext = os.path.splitext(audio_file_path)[1].lower()
                mime = "audio/webm" if ext == ".webm" else "audio/wav"
                files = {"file": (os.path.basename(audio_file_path), audio_file, mime)}

                data = {"model": "whisper-large-v3", "language": language}
                headers = {"Authorization": f"Bearer {self.groq_api_key}"}

                response = requests.post(self.base_url, files=files, data=data, headers=headers, timeout=20)
                response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            transcribed_text = result.get('text', '').strip()
            logger.info(f"Transcription successful: '{transcribed_text[:50]}...'")

            # 🧠 Add LIVE PROMPT VALIDATION here
            from services.prompt_validator import validate_user_input

            intent = validate_user_input(transcribed_text)
            logger.info(f"🧩 Input classified as: {intent}")

            if intent == "emotion":
                logger.info("🎭 Emotion-related input detected — skipping name update.")
                # Just return the text; memory_manager won’t try to treat it as a name
                return transcribed_text  

            return transcribed_text

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Whisper API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during transcription: {e}")
            return None

# Global instance
speech_recognizer = OnlineSpeechRecognizer()
