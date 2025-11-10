#!/usr/bin/env python3
"""
Prompt Validator for SeeForMe
Validates user input intent (name / emotion / other) before memory update.
"""

import re
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
validator_model = genai.GenerativeModel("models/gemini-2.5-flash")

EMOTION_WORDS = {"sad", "happy", "angry", "tired", "depressed", "lonely", "anxious", "neutral", "feeling"}

def local_check(text: str) -> str:
    """Quick local classifier for intent detection."""
    t = text.lower().strip()
    if re.search(r"\b(my name is|this is|i am)\b", t):
        if any(word in t for word in EMOTION_WORDS):
            return "emotion"
        return "name"
    if any(word in t for word in EMOTION_WORDS):
        return "emotion"
    return "other"

def gemini_validate_intent(text: str) -> str:
    """Gemini-based intent check (fallback if unclear)."""
    prompt = f"""
    You are a short text classifier for a voice assistant.
    Classify this text as one of the following:
    - 'name' (if user is telling their name)
    - 'emotion' (if user is expressing feeling or mood)
    - 'other' (if it’s neither)
    Return ONLY ONE WORD.
    Text: "{text}"
    """
    try:
        result = validator_model.generate_content(prompt)
        intent = result.text.strip().lower()
        if intent not in {"name", "emotion", "other"}:
            return local_check(text)
        return intent
    except Exception:
        return local_check(text)

def validate_user_input(text: str) -> str:
    """Combines local + Gemini validation."""
    local_result = local_check(text)
    if local_result != "other":
        return local_result
    return gemini_validate_intent(text)
