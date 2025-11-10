#!/usr/bin/env python3
"""
Web-Optimized Online TTS Handler for SeeForMe
- Uses gTTS for all language support (English, Hindi, Gujarati).
- Caches audio files for repeated phrases to reduce API calls and latency.
- Does NOT play audio; it generates audio files for the web client to play.
"""

import os
import logging
import tempfile
import threading
import hashlib
import shutil
from pathlib import Path
from gtts import gTTS

logger = logging.getLogger(__name__)

class OnlineTTSHandler:
    def __init__(self):
        self.cache_dir = Path("tts_cache")
        self.cache_dir.mkdir(exist_ok=True)
        # Lock to prevent race conditions when creating the same cache file
        self.cache_lock = threading.Lock()
        logger.info("🔊 Online TTS Handler (Web-Optimized) initialized")

    def generate_speech_file(self, text: str, lang: str = 'en') -> str:
        """
        Generates a speech file from text using gTTS and returns its path.
        Uses a cache to avoid re-generating existing audio.
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty.")

        # Create a unique hash for the text and language pair
        text_hash = hashlib.md5(f"{text}_{lang}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{text_hash}.mp3"

        # If the file is already in the cache, return its path immediately.
        if cache_file.exists():
            logger.debug(f"Using cached audio file: {cache_file}")
            return str(cache_file)

        # If not cached, generate a new TTS file (online).
        with self.cache_lock:
            # Double-check if another thread created the file while waiting for the lock
            if cache_file.exists():
                return str(cache_file)
                
            try:
                logger.info(f"Generating new TTS audio for: '{text[:30]}...'")
                tts = gTTS(text=text, lang=lang, slow=False)
                
                # Save to a temporary file first to ensure atomicity
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=self.cache_dir) as tmp_file:
                    tmp_path = tmp_file.name
                    tts.write_to_fp(tmp_file)
                
                # Move the completed file to its final cache location
                shutil.move(tmp_path, str(cache_file))
                
                logger.info(f"Successfully cached TTS audio to {cache_file}")
                return str(cache_file)
                
            except Exception as e:
                logger.error(f"❌ gTTS generation failed: {e}")
                # If generation fails, remove any partial temp file
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise  # Re-raise the exception to be handled by the caller

# Global instance
tts_handler = OnlineTTSHandler()