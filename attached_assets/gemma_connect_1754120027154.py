#!/usr/bin/env python3
"""
Gemma Connector - Integration with Gemma3n:latest LLM via Ollama
Provides intelligent, empathetic responses for vision accessibility
"""

import logging
import requests
import json
import time
import threading
from typing import Dict, Any, Optional
import os
import subprocess
import signal
import sys

logger = logging.getLogger(__name__)

class GemmaConnector:
    """Gemma3n:latest LLM integration for intelligent responses with persistent connection retry"""
    
    def __init__(self):
        self.is_connected = False
        self.model_ready = False
        self.connection_thread = None
        self.shutdown_event = threading.Event()
        
        # Ollama configuration
        self.ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        self.ollama_port = os.getenv('OLLAMA_PORT', '11434')
        self.base_url = f"http://{self.ollama_host}:{self.ollama_port}"
        
        # Fixed model name - only gemma3n:latest
        self.model_name = "gemma3n:latest"
        
        # Connection retry settings
        self.max_connection_attempts = 200  # Keep trying for a long time
        self.base_retry_delay = 5  # Start with 5 seconds
        self.max_retry_delay = 30  # Cap at 30 seconds
        self.current_retry_delay = self.base_retry_delay
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.total_response_time = 0.0
        self.connection_attempts = 0
        
        # Response cache
        self.response_cache = {}
        self.max_cache_size = 100
        
        # Start connection process
        self._start_connection_process()
        
        logger.info("🧠 Gemma3n connector initialized - attempting connection...")
    
    def _start_connection_process(self):
        """Start background connection process"""
        self.connection_thread = threading.Thread(
            target=self._connection_loop, 
            daemon=True,
            name="Gemma3nConnector"
        )
        self.connection_thread.start()
    
    def _connection_loop(self):
        """Continuously attempt to connect to Gemma3n:latest until successful"""
        print("🔍 Starting Gemma3n:latest connection process...")
        
        while not self.is_connected and not self.shutdown_event.is_set():
            try:
                self.connection_attempts += 1
                print(f"🔄 Gemma3n:latest connection attempt #{self.connection_attempts}")
                
                # Step 1: Check if Ollama server is running
                if not self._check_ollama_server():
                    self._attempt_start_ollama()
                    time.sleep(5)  # Give server time to start
                    continue
                
                # Step 2: Check if gemma3n:latest model is available
                if not self._check_model_availability():
                    print(f"❌ Model {self.model_name} not found")
                    print("💡 Please make sure you have downloaded the model:")
                    print(f"💡 Run: ollama pull {self.model_name}")
                    self._wait_before_retry()
                    continue
                
                # Step 3: Test the model
                if self._test_model():
                    self.model_ready = True
                    self.is_connected = True
                    self.current_retry_delay = self.base_retry_delay  # Reset delay
                    print(f"✅ Successfully connected to {self.model_name}!")
                    print(f"🎉 Ready to provide intelligent responses!")
                    return
                else:
                    print(f"❌ Model {self.model_name} failed testing")
                    
            except Exception as e:
                print(f"❌ Connection attempt #{self.connection_attempts} failed: {e}")
            
            # Wait before retry with exponential backoff
            self._wait_before_retry()
        
        if not self.is_connected:
            print("⚠️ Gemma3n connection process ended - using fallback responses")
    
    def _check_ollama_server(self):
        """Check if Ollama server is running and accessible"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama server is running")
                return True
            else:
                print(f"⚠️ Ollama server responded with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("⚠️ Cannot connect to Ollama server - is it running?")
            print("💡 Please start Ollama: 'ollama serve'")
            return False
        except Exception as e:
            print(f"⚠️ Ollama server check failed: {e}")
            return False
    
    def _attempt_start_ollama(self):
        """Attempt to start Ollama server if possible"""
        try:
            # Check if ollama command is available
            result = subprocess.run(['which', 'ollama'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("🚀 Attempting to start Ollama server...")
                # Start ollama serve in background
                subprocess.Popen(['ollama', 'serve'], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
                print("✅ Ollama server start command executed")
                return True
            else:
                print("💡 Ollama not found in PATH. Please start manually: 'ollama serve'")
                return False
        except Exception as e:
            print(f"⚠️ Could not start Ollama server: {e}")
            print("💡 Please start Ollama manually: 'ollama serve'")
            return False
    
    def _check_model_availability(self):
        """Check if gemma3n:latest model is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                available_models = [model['name'] for model in data.get('models', [])]
                print(f"📋 Available models: {available_models}")
                
                # Check for exact match or partial match
                for available_model in available_models:
                    if (self.model_name == available_model or 
                        available_model.startswith('gemma3n') or
                        'gemma3n' in available_model.lower()):
                        print(f"✅ Found model: {available_model}")
                        # Update model name to exact match
                        self.model_name = available_model
                        return True
                
                print(f"❌ Model {self.model_name} not found in available models")
                return False
            else:
                print(f"❌ Failed to get model list: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error checking model availability: {e}")
            return False
    
    def _test_model(self):
        """Test if the gemma3n:latest model responds correctly"""
        try:
            test_prompt = "Hello! Please respond with exactly: 'I am Gemma3n and I am ready to help!'"
            
            print(f"🧪 Testing {self.model_name} with simple prompt...")
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "max_tokens": 30,
                        "timeout": 45
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                if response_text and len(response_text) > 0:
                    print(f"✅ Model test successful!")
                    print(f"📝 Response: '{response_text}'")
                    return True
                else:
                    print("❌ Model returned empty response")
                    return False
            else:
                print(f"❌ Model test failed with status {response.status_code}")
                if response.text:
                    print(f"Error details: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Model test timed out - model might be loading")
            return False
        except Exception as e:
            print(f"❌ Model test error: {e}")
            return False
    
    def _wait_before_retry(self):
        """Wait before next retry with exponential backoff"""
        print(f"⏳ Waiting {self.current_retry_delay} seconds before next attempt...")
        print(f"💡 Make sure {self.model_name} is downloaded: ollama pull {self.model_name}")
        
        # Wait with ability to interrupt
        for i in range(self.current_retry_delay):
            if self.shutdown_event.is_set():
                return
            remaining = self.current_retry_delay - i
            if remaining % 5 == 0 or remaining <= 3:
                print(f"⏳ Retrying in {remaining} seconds...")
            time.sleep(1)
        
        # Exponential backoff
        self.current_retry_delay = min(self.current_retry_delay * 1.2, self.max_retry_delay)
    
    def generate_vision_response(self, context: Dict[str, Any]) -> str:
        """Generate response for vision-based queries"""
        if not self.is_connected or not self.model_ready:
            return self._get_fallback_vision_response(context)
        
        try:
            # Build vision-specific prompt
            prompt = self._build_vision_prompt(context)
            
            # Generate response
            response = self._call_gemma(prompt)
            return response if response else self._get_fallback_vision_response(context)
            
        except Exception as e:
            logger.error(f"❌ Vision response generation failed: {e}")
            return self._get_fallback_vision_response(context)
    
    def generate_text_response(self, context: Dict[str, Any]) -> str:
        """Generate response for text-only queries"""
        if not self.is_connected or not self.model_ready:
            return self._get_fallback_text_response(context)
        
        try:
            # Build text-specific prompt
            prompt = self._build_text_prompt(context)
            
            # Generate response
            response = self._call_gemma(prompt)
            return response if response else self._get_fallback_text_response(context)
            
        except Exception as e:
            logger.error(f"❌ Text response generation failed: {e}")
            return self._get_fallback_text_response(context)
    
    def _build_vision_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for vision-based responses"""
        user_input = context.get('user_input', '')
        user_context = context.get('user_context', {})
        vision_results = context.get('vision_results', {})
        
        # System prompt for vision assistant
        system_prompt = """You are SeeForMe, an empathetic AI vision assistant for blind and visually impaired users. 

Your role is to:
1. Describe visual information clearly and helpfully
2. Provide emotional support when users share feelings
3. Be concise but caring (under 50 words typically)
4. Use "I can see" when describing visual information
5. Acknowledge emotions with empathy

Guidelines:
- Be warm, supportive, and encouraging
- Focus on practical, useful information
- Use simple, clear language
- Show understanding of accessibility needs"""

        # Build context information
        context_parts = []
        
        # User information
        user_name = user_context.get('name', 'friend')
        current_emotion = user_context.get('current_emotion', 'neutral')
        context_parts.append(f"User's name: {user_name}")
        context_parts.append(f"User's current emotion: {current_emotion}")
        
        # Vision analysis results
        if 'scene' in vision_results:
            scene = vision_results['scene']
            scene_type = scene.get('scene_type', 'unknown')
            objects = scene.get('objects', [])
            people_count = scene.get('people_count', 0)
            
            context_parts.append(f"Scene: {scene_type}")
            if objects:
                context_parts.append(f"Objects visible: {', '.join(objects[:5])}")
            if people_count > 0:
                context_parts.append(f"People count: {people_count}")
        
        if 'emotion' in vision_results:
            emotion_data = vision_results['emotion']
            detected_emotion = emotion_data.get('emotion', 'neutral')
            confidence = emotion_data.get('confidence', 0.0)
            context_parts.append(f"User's facial expression: {detected_emotion} (confidence: {confidence:.2f})")
        
        # Build final prompt
        context_str = '\n'.join(context_parts)
        
        prompt = f"""{system_prompt}

Context:
{context_str}

User asked: "{user_input}"

Respond naturally and helpfully:"""
        
        return prompt
    
    def _build_text_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for text-only responses"""
        user_input = context.get('user_input', '')
        user_context = context.get('user_context', {})
        intent = context.get('intent', 'general_conversation')
        
        # System prompt for text assistant
        system_prompt = """You are SeeForMe, an empathetic AI assistant for blind and visually impaired users.

Your role is to:
1. Provide emotional support and encouragement
2. Help with accessibility-related questions
3. Be a friendly, understanding companion
4. Keep responses concise but caring (under 50 words typically)
5. Show empathy and understanding

Guidelines:
- Be warm, supportive, and encouraging
- Use simple, clear language
- Acknowledge feelings and provide comfort when needed
- Offer practical help and suggestions"""

        # Build context
        user_name = user_context.get('name', 'friend')
        current_emotion = user_context.get('current_emotion', 'neutral')
        
        context_str = f"""User's name: {user_name}
Current emotion: {current_emotion}
Intent: {intent}"""
        
        # Recent conversation
        history = user_context.get('conversation_history', [])
        if history:
            recent_history = history[-2:]  # Last 2 exchanges
            history_str = '\n'.join([
                f"{'User' if 'user' in entry else 'Assistant'}: {list(entry.values())[0]}"
                for entry in recent_history
            ])
            context_str += f"\n\nRecent conversation:\n{history_str}"
        
        prompt = f"""{system_prompt}

Context:
{context_str}

User said: "{user_input}"

Respond naturally and helpfully:"""
        
        return prompt
    
    def _call_gemma(self, prompt: str) -> Optional[str]:
        """Call Gemma3n API with prompt"""
        if not self.is_connected or not self.model_name:
            return None
            
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = hash(prompt) % 1000000  # Simple hash for caching
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Make API call
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 150,
                        "stop": ["\n\nUser:", "\n\nHuman:", "User said:", "\n\n"]
                    }
                },
                timeout=45
            )
            
            response_time = time.time() - start_time
            self.total_response_time += response_time
            self.request_count += 1
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                if generated_text:
                    # Clean response
                    cleaned = self._clean_response(generated_text)
                    
                    # Cache response
                    if len(self.response_cache) < self.max_cache_size:
                        self.response_cache[cache_key] = cleaned
                    
                    self.success_count += 1
                    print(f"✅ Gemma3n response generated in {response_time:.2f}s")
                    return cleaned
                else:
                    print("⚠️ Gemma3n returned empty response")
                
            else:
                print(f"❌ Gemma3n API error: {response.status_code}")
                if response.text:
                    print(f"Error details: {response.text}")
                # Connection might be lost, trigger reconnection
                if response.status_code in [502, 503, 504]:
                    self.is_connected = False
                    self._start_connection_process()
                
        except requests.exceptions.Timeout:
            print("⏱️ Gemma3n request timed out")
        except requests.exceptions.ConnectionError:
            print("🔌 Gemma3n connection lost - attempting reconnection")
            self.is_connected = False
            self._start_connection_process()
        except Exception as e:
            print(f"❌ Gemma3n API call failed: {e}")
        
        return None
    
    def _clean_response(self, response: str) -> str:
        """Clean generated response"""
        # Remove common prefixes
        prefixes = ["Assistant:", "SeeForMe:", "Response:", "AI:", "Gemma3n:", "Gemma:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Remove quotes if the entire response is quoted
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Remove asterisks and action markers
        response = response.replace('*', '').strip()
        
        # Ensure proper ending
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
    
    def _get_fallback_vision_response(self, context: Dict[str, Any]) -> str:
        """Fallback response for vision queries"""
        vision_results = context.get('vision_results', {})
        
        if 'scene' in vision_results:
            scene = vision_results['scene']
            scene_type = scene.get('scene_type', 'an area')
            objects = scene.get('objects', [])
            people_count = scene.get('people_count', 0)
            
            parts = [f"I can see you're in {scene_type}."]
            
            if people_count > 0:
                if people_count == 1:
                    parts.append("There's one person visible.")
                else:
                    parts.append(f"I can see {people_count} people.")
            
            if objects:
                if len(objects) <= 2:
                    parts.append(f"I can see {' and '.join(objects)}.")
                else:
                    parts.append(f"I can see {objects[0]}, {objects[1]}, and other items.")
            
            return ' '.join(parts)
        
        elif 'emotion' in vision_results:
            emotion_data = vision_results['emotion']
            emotion = emotion_data.get('emotion', 'neutral')
            confidence = emotion_data.get('confidence', 0.0)
            
            if confidence > 0.6:
                return f"You look {emotion.lower()}. I'm here to support you."
            else:
                return "I can see your face, but I'm not completely certain about your expression right now."
        
        return "I'm analyzing what I can see to help you better."
    
    def _get_fallback_text_response(self, context: Dict[str, Any]) -> str:
        """Fallback response for text queries - provides immediate responses"""
        user_input = context.get('user_input', '').lower()
        user_name = context.get('user_context', {}).get('name', 'friend')
        intent = context.get('intent', 'general_conversation')
        
        # Always provide immediate, engaging responses
        if 'hello' in user_input or 'hi' in user_input or 'hey' in user_input:
            return f"Hello {user_name}! I'm SeeForMe, your AI assistant. I can help you understand your surroundings, detect emotions, and provide support. What would you like me to help you with?"
        
        if 'name' in user_input and ('my' in user_input or 'i am' in user_input or 'i\'m' in user_input):
            return f"Nice to meet you, {user_name}! I'm excited to be your AI companion. I can see your surroundings, analyze emotions, and have conversations with you."
        
        # Emotion-based responses
        if any(word in user_input for word in ['sad', 'upset', 'angry', 'frustrated', 'worried', 'depressed', 'down']):
            return f"I can hear that you might be going through something difficult, {user_name}. I'm here to listen and support you. Would you like me to look at your expression to better understand how you're feeling?"
        
        if any(word in user_input for word in ['happy', 'excited', 'good', 'great', 'wonderful', 'amazing', 'fantastic']):
            return f"That's wonderful to hear, {user_name}! I'm so glad you're feeling positive. Your happiness makes me happy too!"
        
        # Scene/vision requests
        if any(phrase in user_input for phrase in ['what do you see', 'what\'s there', 'look around', 'describe', 'where am i']):
            return f"I'd love to help you see your surroundings, {user_name}! Let me switch to the back camera and analyze what's around you."
        
        # Emotion/self requests  
        if any(phrase in user_input for phrase in ['how do i look', 'my expression', 'my face', 'my mood', 'how am i']):
            return f"I can help you understand your current expression and mood, {user_name}. Let me look at your face using the front camera."
        
        # Help requests
        if any(word in user_input for word in ['help', 'assist', 'support', 'guide']):
            return f"I'm here to help you, {user_name}! I can describe your surroundings, analyze your emotions, have conversations, and provide support. Just tell me what you need!"
        
        # General conversation responses
        responses = [
            f"I'm listening, {user_name}. Tell me more about what's on your mind.",
            f"That's interesting, {user_name}. I'm here to chat and help however I can.",
            f"Thanks for sharing with me, {user_name}. How can I best support you right now?",
            f"I appreciate you talking with me, {user_name}. What would you like to discuss or explore together?",
            f"I understand, {user_name}. I'm here to support you in whatever way you need."
        ]
        
        import random
        return random.choice(responses)
    
    def test_connection(self) -> bool:
        """Test current connection status"""
        if not self.is_connected:
            return False
            
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status"""
        success_rate = (self.success_count / self.request_count * 100) if self.request_count > 0 else 0
        avg_response_time = (self.total_response_time / self.request_count) if self.request_count > 0 else 0
        
        return {
            'status': 'ready' if (self.is_connected and self.model_ready) else 'connecting' if self.connection_thread and self.connection_thread.is_alive() else 'failed',
            'connected': self.is_connected,
            'model_ready': self.model_ready,
            'model_name': self.model_name,
            'connection_attempts': self.connection_attempts,
            'total_requests': self.request_count,
            'success_rate': f"{success_rate:.1f}%",
            'avg_response_time': f"{avg_response_time:.2f}s",
            'cache_size': len(self.response_cache),
            'base_url': self.base_url
        }
    
    def force_reconnect(self):
        """Force a reconnection attempt"""
        print("🔄 Forcing Gemma3n reconnection...")
        self.is_connected = False
        self.model_ready = False
        self.current_retry_delay = self.base_retry_delay
        
        if not self.connection_thread or not self.connection_thread.is_alive():
            self._start_connection_process()
    
    def wait_for_connection(self, timeout: int = 120) -> bool:
        """Wait for connection to be established"""
        start_time = time.time()
        print(f"⏳ Waiting up to {timeout} seconds for Gemma3n:latest connection...")
        
        while not self.is_connected and (time.time() - start_time) < timeout:
            time.sleep(1)
            if self.shutdown_event.is_set():
                return False
        
        if self.is_connected:
            print(f"✅ Gemma3n connected after {time.time() - start_time:.1f} seconds")
            return True
        else:
            print(f"⏱️ Gemma3n connection timeout after {timeout} seconds")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up Gemma3n connector...")
        self.shutdown_event.set()
        
        # Wait for connection thread to finish
        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=2)
        
        # Close session
        try:
            self.session.close()
        except Exception:
            pass
        
        # Clear cache
        self.response_cache.clear()
        
        print("✅ Gemma3n connector cleanup completed")


# Convenience functions for easy integration
def create_gemma_connector() -> GemmaConnector:
    """Create and return a new Gemma3n connector instance"""
    return GemmaConnector()

def test_gemma_availability() -> bool:
    """Quick test to check if Gemma3n is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return any('gemma3n' in model.lower() for model in models)
        return False
    except Exception:
        return False

# Example usage and testing
if __name__ == "__main__":
    import signal
    import sys
    
    def signal_handler(signum, frame):
        print("\n🛑 Shutdown signal received")
        if 'connector' in locals():
            connector.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🧠 Testing Gemma3n:latest Connector...")
    
    # Create connector
    connector = GemmaConnector()
    
    # Wait for connection
    if connector.wait_for_connection(timeout=180):
        print("🎉 Connection successful! Testing responses...")
        
        # Test text response
        test_context = {
            'user_input': 'Hello, how are you?',
            'user_context': {'name': 'Test User', 'current_emotion': 'neutral'},
            'intent': 'greeting'
        }
        
        response = connector.generate_text_response(test_context)
        print(f"💬 Test response: {response}")
        
        # Show status
        status = connector.get_status()
        print(f"📊 Status: {status}")
        
        print("✅ All tests completed successfully!")
    else:
        print("❌ Connection failed - check your setup:")
        print("💡 1. Make sure Ollama is running: ollama serve")
        print("💡 2. Make sure gemma3n:latest is installed: ollama pull gemma3n:latest")
        print("💡 3. Check if model exists: ollama list")
    
    # Cleanup
    connector.cleanup()