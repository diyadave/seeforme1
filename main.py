import os
import base64
import tempfile
import asyncio
import logging
from datetime import datetime

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

from services.speech_handler import OnlineSpeechRecognizer as SpeechHandler
from services.tts_handler import OnlineTTSHandler as TTSHandler 
from services.memory_manager import offline_memory_manager
from services.vision_processor import vision_processor
from services.gemma_connector import gemini_connector

# Load environment variables
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Access your API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure Gemini
if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

# Create a global event loop for the entire application
global_loop = asyncio.new_event_loop()
asyncio.set_event_loop(global_loop)

# Init Flask + SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'seeforme_secret_key_2024')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Handlers
speech_recognizer = SpeechHandler()
tts_handler = TTSHandler()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store client states
client_states = {}

@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    client_id = request.sid
    client_states[client_id] = {
        'last_interaction': datetime.now(),
        'user_name': None,
        'camera_active': False
    }
    logger.info(f"✅ Client {client_id} connected")
    emit("server_status", {"message": "Connected to SeeForMe Assistant", "status": "connected"})


@socketio.on("disconnect")
def handle_disconnect():
    client_id = request.sid
    if client_id in client_states:
        del client_states[client_id]
    logger.info(f"❌ Client {client_id} disconnected")


def run_async_safe(coro):
    """Helper function to run async code safely in sync context"""
    try:
        # Try to use the global loop first
        if global_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, global_loop)
            return future.result(timeout=30)  # 30 second timeout
        else:
            # If global loop is not running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
    except Exception as e:
        logger.error(f"Async operation failed: {e}")
        return None


@socketio.on("user_audio")
def handle_user_audio(data):
    """Handles audio input from frontend"""
    client_id = request.sid
    try:
        audio_b64 = data.get("audio_data")
        language = data.get("language", "en")
        audio_format = data.get("format", "audio/webm")

        if not audio_b64:
            emit("error", {"message": "No audio data received"})
            return

        logger.info(f"Received audio data. Length: {len(audio_b64)}, Format: {audio_format}")

        # Decode audio
        try:
            audio_bytes = base64.b64decode(audio_b64)
            logger.info(f"Decoded audio bytes: {len(audio_bytes)}")
        except Exception as e:
            logger.error(f"Failed to decode base64 audio: {e}")
            emit("error", {"message": "Invalid audio data"})
            return

        if len(audio_bytes) == 0:
            emit("error", {"message": "Empty audio data"})
            return

        # Determine file extension based on format
        if "webm" in audio_format:
            file_ext = ".webm"
        elif "mp4" in audio_format:
            file_ext = ".mp4"
        elif "wav" in audio_format:
            file_ext = ".wav"
        else:
            file_ext = ".webm"  # default

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        logger.info(f"Saved audio to temp file: {temp_audio_path}")

        # 1. Transcribe speech → text
        user_text = speech_recognizer.transcribe_audio_file(temp_audio_path, language)
        
        # Clean up temp file
        try:
            os.unlink(temp_audio_path)
        except:
            pass
        
        if not user_text or user_text.strip() == "":
            logger.warning("Transcription returned empty or None")
            emit("response_text", {"text": "I couldn't hear that clearly. Please try again."})
            return

        logger.info(f"Transcription successful: '{user_text}'")
        emit("transcription_result", {"text": user_text})

        # Check if user wants scene description
        vision_keywords = ["what do you see", "describe", "scene", "look", "camera", 
                          "vision", "what's in front", "what am i looking at", "what is this",
                          "identify", "recognize", "see anything"]
        
        if any(keyword in user_text.lower() for keyword in vision_keywords):
            # Request scene frame from frontend
            emit("request_scene_frame", {
                "user_text": user_text, 
                "language": language,
                "request_id": f"vision_{datetime.now().timestamp()}"
            })
            return

        # Get user context and build conversation prompt
        user_name = offline_memory_manager.process_name_learning(user_text)
        if not user_name:
            user_name = offline_memory_manager.get_last_user_name() or "friend"

        # Update client state
        if client_id in client_states:
            client_states[client_id]['user_name'] = user_name
            client_states[client_id]['last_interaction'] = datetime.now()

        # Get emotional continuity if available
        continuity_prompt = offline_memory_manager.get_emotional_continuity_prompt()

        # Build context for Gemini
        context = {
            'user_context': {
                'name': user_name,
                'current_emotion': 'neutral',
                'continuity_prompt': continuity_prompt
            },
            'user_input': user_text,
            'vision_results': None
        }

        # 2. Generate response using Gemini connector (FIXED)
        ai_reply = run_async_safe(gemini_connector.generate_response(context))
        
        if not ai_reply:
            ai_reply = "I'm sorry, I'm having trouble thinking right now. Could you try again?"

        logger.info(f"Gemini reply: {ai_reply}")

        # Save conversation to memory
        offline_memory_manager.save_conversation(user_name, user_text, ai_reply, 'neutral')

        # 3. Convert AI reply → speech
        try:
            speech_file = tts_handler.generate_speech_file(ai_reply, language)
            with open(speech_file, "rb") as f:
                audio_bytes = f.read()
            audio_b64_out = base64.b64encode(audio_bytes).decode("utf-8")

            # 4. Send back to frontend
            emit("response_audio", {
                "text": ai_reply,
                "audio_data": audio_b64_out,
                "format": "audio/mp3"
            })
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            # Send text-only response
            emit("response_text", {"text": ai_reply})

    except Exception as e:
        logger.error(f"Error in handle_user_audio: {e}")
        emit("error", {"message": "Sorry, I encountered an error processing your message."})


@socketio.on("vision_input")
def handle_vision_input(data):
    """Handles vision analysis requests from frontend"""
    client_id = request.sid
    try:
        image_data = data.get("image_data")
        vision_type = data.get("type", "scene_description")
        user_text = data.get("user_text", "What do you see?")
        language = data.get("language", "en")

        if not image_data:
            emit("error", {"message": "No image data received"})
            return

        # Convert data URL to bytes
        if image_data.startswith('data:image'):
            header, image_data = image_data.split(',', 1)
        
        image_bytes = base64.b64decode(image_data)

        logger.info(f"Processing vision input: {vision_type}")

        # Analyze scene
        if vision_type == "scene_description":
            vision_result = vision_processor.analyze_scene(image_bytes)
        elif vision_type == "emotion_detection":
            vision_result = vision_processor.detect_emotion(image_bytes)
        else:
            vision_result = {'status': 'error', 'description': 'Unknown vision type.'}


        if vision_result.get('status') == 'error':
            emit("response_text", {"text": vision_result.get('description', 'Vision analysis failed.')})
            return

        # Get user context
        user_name = offline_memory_manager.get_last_user_name() or "friend"
        if client_id in client_states and client_states[client_id]['user_name']:
            user_name = client_states[client_id]['user_name']
        
        # Build context with vision results
        context = {
            'user_context': {
                'name': user_name,
                'current_emotion': 'neutral',
                'continuity_prompt': ''
            },
            'user_input': user_text,
            'vision_results': vision_result
        }

        # Generate contextual response (FIXED)
        ai_reply = run_async_safe(gemini_connector.generate_response(context))
        
        if not ai_reply:
            ai_reply = vision_result.get('description', 'I can see the scene but having trouble describing it.')

        logger.info(f"Vision response: {ai_reply}")

        # Save conversation
        offline_memory_manager.save_conversation(user_name, user_text, ai_reply, 'neutral')

        # Generate speech response
        try:
            speech_file = tts_handler.generate_speech_file(ai_reply, language)
            with open(speech_file, "rb") as f:
                audio_bytes = f.read()
            audio_b64_out = base64.b64encode(audio_bytes).decode("utf-8")

            emit("response_audio", {
                "text": ai_reply,
                "audio_data": audio_b64_out,
                "format": "audio/mp3"
            })
        except Exception as e:
            logger.error(f"TTS generation failed for vision response: {e}")
            emit("response_text", {"text": ai_reply})

    except Exception as e:
        logger.error(f"Error in handle_vision_input: {e}")
        emit("error", {"message": "Sorry, I had trouble analyzing the image."})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting SeeForMe server on port {port}")
    
    # Start the global event loop in a separate thread
    def run_loop():
        asyncio.set_event_loop(global_loop)
        global_loop.run_forever()
    
    import threading
    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    
    socketio.run(app, host="0.0.0.0", port=port, debug=True)