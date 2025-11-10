#!/usr/bin/env python3
import os
import logging
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()
logger = logging.getLogger(__name__)

class GeminiConnector:
    def __init__(self):
        self.is_connected = False
        self.model = None
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._connect()
        logger.info("🚀 Gemini API Connector initialized")

    def _connect(self):
        """Initialize the connection to the Gemini API."""
        if not self.api_key:
            logger.error("❌ GEMINI_API_KEY not found. Gemini connector disabled.")
            return

        try:
            genai.configure(api_key=self.api_key)
            # ✅ Use the correct model name (no 'models/' prefix)
            self.model = genai.GenerativeModel("models/gemini-2.5-flash")

            # Quick test call to confirm connectivity
            test = self.model.generate_content("hello")
            if test.text:
                logger.info("✅ Connected to Gemini API successfully.")
                self.is_connected = True
            else:
                logger.warning("⚠️ Gemini test response empty; connection uncertain.")
        except Exception as e:
            logger.error(f"❌ Gemini API connection failed: {e}")
            self.is_connected = False

    async def generate_response(self, context: Dict[str, Any]) -> str:
        """Generate a contextual Gemini response."""
        if not self.is_connected or not self.model:
            return "I'm having trouble connecting to my brain right now. Please try again later."

        try:
            prompt = self._build_prompt(context)
            # Call the async version correctly
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"❌ Gemini generation failed: {e}")
            return "I'm sorry, I encountered an error while thinking. Could you rephrase that?"

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build a contextual prompt for Gemini."""
        user_context = context.get("user_context", {})
        name = user_context.get("name", "friend")
        emotion = user_context.get("current_emotion", "neutral")
        vision = context.get("vision_results")
        user_input = context.get("user_input", "...")
        continuity = user_context.get("continuity_prompt", "")

        prompt = f"You are SeeForMe, an empathetic AI assistant for a visually impaired person named {name}.\n"
        if continuity:
            prompt += continuity + "\n"
        prompt += f"Their current detected emotion is '{emotion}'.\n"
        if vision and 'description' in vision:
            prompt += f"You see: \"{vision['description']}\"\n"
        prompt += f"The user says: \"{user_input}\"\n\n"
        prompt += "Respond warmly in under 50 words."
        return prompt


# Global instance
gemini_connector = GeminiConnector()
