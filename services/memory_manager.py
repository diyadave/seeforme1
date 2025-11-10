#!/usr/bin/env python3
"""
Offline Memory Manager for SeeForMe
- Stores conversations and emotional states in simple JSON files.
- Remembers user's name and last significant emotion between sessions.
"""

import logging
import time
import re
import os
import json
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class OfflineMemoryManager:
    def __init__(self, memory_dir="memory_data"):
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)
        
        self.user_profile_file = os.path.join(self.memory_dir, "user_profile.json")
        self.conversation_file = os.path.join(self.memory_dir, "conversations.json")
        
        self.user_profile = self._load_json(
            self.user_profile_file, 
            default={"name": None, "last_emotion": None}
        )
        self.conversations = self._load_json(self.conversation_file, default=[])
        
        # Common non-names we should ignore
        self.invalid_names = {
            "sad", "happy", "angry", "upset", "tired", "depressed", 
            "alone", "good", "bad", "okay", "fine", "feeling"
        }

        logger.info("💾 Offline Memory Manager initialized")

    def _load_json(self, file_path, default=None):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def _save_json(self, file_path, data):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def process_name_learning(self, user_text: str) -> Optional[str]:
        """Extracts and learns the user's name from text safely (with live validation)."""
        from services.prompt_validator import validate_user_input  # ✅ import locally to avoid circular imports
        user_text_lower = user_text.lower().strip()

        # 🧠 Step 1: classify the input intent first
        intent = validate_user_input(user_text)
        if intent == "emotion":
            logger.info("🎭 Emotion-related input detected — skipping name learning.")
            return self.user_profile.get("name")

        # 🧩 Step 2: extract a name only if intent == "name"
        if intent == "name":
            name_patterns = [
                r"\bmy name is\s+([A-Z][a-z]+)\b",
                r"\bcall me\s+([A-Z][a-z]+)\b",
                r"\bi am\s+([A-Z][a-z]+)\b",
                r"\bi'm\s+([A-Z][a-z]+)\b"
            ]

            for pattern in name_patterns:
                match = re.search(pattern, user_text, re.IGNORECASE)
                if match:
                    candidate = match.group(1).capitalize()
                    if candidate.lower() not in self.invalid_names:
                        if self.user_profile.get("name") != candidate:
                            self.user_profile["name"] = candidate
                            self._save_json(self.user_profile_file, self.user_profile)
                            logger.info(f"📝 Learned new user name: {candidate}")
                        return candidate

            logger.info("⚠️ No valid name detected even though intent was 'name'.")
            return self.user_profile.get("name")

        # 🪞 Step 3: detect other people's names only if needed
        person_name_patterns = [
            r"\bher name is\s+([A-Z][a-z]+)\b",
            r"\bhis name is\s+([A-Z][a-z]+)\b",
            r"\btheir name is\s+([A-Z][a-z]+)\b",
            r"\bthat is\s+([A-Z][a-z]+)\b"
        ]

        for pattern in person_name_patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                person_name = match.group(1).capitalize()
                if person_name.lower() not in self.invalid_names:
                    self.user_profile["last_seen_person_name"] = person_name
                    self._save_json(self.user_profile_file, self.user_profile)
                    logger.info(f"👤 Learned name of person in scene: {person_name}")

        # 🧩 Step 4: return last remembered valid name
        return self.user_profile.get("name")


    def get_last_user_name(self) -> Optional[str]:
        return self.user_profile.get("name")

    def save_conversation(self, user_name: str, user_input: str, ai_response: str, emotion: str):
        """Saves a conversation entry."""
        entry = {
            'timestamp': time.time(),
            'user_name': user_name,
            'user_input': user_input,
            'ai_response': ai_response,
            'emotion': emotion
        }
        self.conversations.append(entry)
        # Keep only the last 50 conversations
        self.conversations = self.conversations[-50:]
        self._save_json(self.conversation_file, self.conversations)

    def save_last_emotion(self, emotion: str):
        """Saves the last significant emotion of the user."""
        if emotion and emotion not in ["neutral", "error"]:
            self.user_profile["last_emotion"] = {
                "emotion": emotion,
                "timestamp": time.time()
            }
            self._save_json(self.user_profile_file, self.user_profile)
            logger.info(f"💭 Saved significant emotion '{emotion}' for user.")

    def get_emotional_continuity_prompt(self) -> str:
        """
        If a significant emotion was saved recently, returns a prompt for the AI.
        """
        last_emotion_data = self.user_profile.get("last_emotion")
        if not last_emotion_data:
            return ""

        # Check if the emotion was recorded in the last 24 hours (86400 seconds)
        time_since = time.time() - last_emotion_data.get("timestamp", 0)
        if 0 < time_since < 86400:
            emotion = last_emotion_data["emotion"]
            logger.info(f"Found recent emotion '{emotion}'. Generating continuity prompt.")
            # Clear the emotion after using it once to avoid repetition
            self.user_profile["last_emotion"] = None
            self._save_json(self.user_profile_file, self.user_profile)
            return (
                f"SYSTEM NOTE: The user was feeling '{emotion}' in their last session. "
                f"Start the conversation by gently acknowledging this and asking how they are doing now."
            )
        
        return ""

# Global instance for easy import
offline_memory_manager = OfflineMemoryManager()
