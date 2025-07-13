import streamlit as st
import tempfile
import os
import requests
import json
from pathlib import Path
import time

# Text-to-Speech imports
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    st.warning("pyttsx3 not installed. TTS will be disabled.")

# Audio recording imports
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False
    st.warning("Audio recording not available.")


class OllamaLLM:
    """Interface for Ollama API"""
    
    def __init__(self, model_name: str = "phi3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.chat_url = f"{base_url}/api/chat"
        
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except:
            pass
        return []
    
    def chat(self, message: str, conversation_history: list = None) -> str:
        """Send message to Ollama and get response"""
        try:
            messages = conversation_history or []
            messages.append({"role": "user", "content": message})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False
            }
            
            response = requests.post(self.chat_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('message', {}).get('content', 'No response received')
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return "Error: Request timed out. Ollama might be busy."
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"


class TextToSpeech:
    """Text-to-Speech handler"""
    
    def __init__(self):
        self.engine = None
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self._setup_voice()
            except:
                self.engine = None
    
    def _setup_voice(self):
        """Setup TTS voice properties"""
        if self.engine:
            voices = self.engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    self.engine.setProperty('voice', voices[0].id)
            
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.8)
    
    def speak(self, text: str) -> bool:
        """Convert text to speech"""
        if not self.engine:
            return False
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except:
            return False
    
    def is_available(self) -> bool:
        """Check if TTS is available"""
        return self.engine is not None


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'llm' not in st.session_state:
        st.session_state.llm = OllamaLLM(model_name="phi3")
    
    if 'tts' not in st.session_state:
        st.session_state.tts = TextToSpeech()


def display_conversation():
    """Display conversation history"""
    st.subheader("💬 Conversation History")
    
    if not st.session_state.conversation_history:
        st.info("No conversation yet. Start by typing a message!")
        return
    
    for i, msg in enumerate(st.session_state.conversation_history):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        if role == 'user':
            with st.chat_message("user"):
                st.write(f"👤 **You:** {content}")
        elif role == 'assistant':
            with st.chat_message("assistant"):
                st.write(f"🤖 **Assistant:** {content}")
                
                # Add TTS button for assistant responses
                if st.session_state.tts.is_available():
                    if st.button(f"🔊 Play Response {i}", key=f"tts_{i}"):
                        st.session_state.tts.speak(content)


def main():
    st.set_page_config(
        page_title="Simple Chatbot",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 Simple Text Chatbot (No ASR)")
    st.markdown("*Chat with AI using text input - ASR functionality disabled due to PyTorch issues*")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check system status
        st.subheader("System Status")
        
        # ASR Status
        st.write("🎯 **ASR:** ❌ Disabled (PyTorch issues)")
        
        # Ollama Status
        if st.session_state.llm.is_available():
            available_models = st.session_state.llm.get_available_models()
            st.write("🧠 **Ollama:** ✅ Connected")
            
            if available_models:
                selected_model = st.selectbox(
                    "Select Model:",
                    available_models,
                    index=available_models.index("phi3") if "phi3" in available_models else 0
                )
                st.session_state.llm.model_name = selected_model
        else:
            st.write("🧠 **Ollama:** ❌ Not connected")
            st.error("Please start Ollama server: `ollama serve`")
        
        # TTS Status
        if st.session_state.tts.is_available():
            st.write("🔊 **TTS:** ✅ Available")
        else:
            st.write("🔊 **TTS:** ❌ Not available")
        
        st.divider()
        
        # PyTorch fix instructions
        st.subheader("🔧 Fix PyTorch Issues")
        st.markdown("""
        **To enable ASR:**
        1. Run: `python fix_pytorch.py`
        2. Install VC++ Redistributable
        3. Restart computer
        4. Use full voice_chatbot.py
        """)
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display conversation
        display_conversation()
    
    with col2:
        st.subheader("💬 Text Input")
        
        # Text input
        text_input = st.text_area("Type your message:", height=150, key="text_input")
        
        if st.button("💬 Send Message", type="primary"):
            if text_input.strip():
                # Add to conversation
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": text_input
                })
                
                # Get LLM response
                if st.session_state.llm.is_available():
                    with st.spinner("Getting AI response..."):
                        response = st.session_state.llm.chat(
                            text_input, 
                            st.session_state.conversation_history[:-1]
                        )
                    
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Auto-play response if TTS is available
                    if st.session_state.tts.is_available():
                        st.session_state.tts.speak(response)
                    
                    # Clear input
                    st.rerun()
                else:
                    st.error("Ollama not available for response generation")
            else:
                st.warning("Please enter a message")
        
        st.divider()
        
        # Instructions
        st.subheader("📋 Instructions")
        st.markdown("""
        **Current Status:** Text-only chatbot
        
        **To enable voice features:**
        1. Fix PyTorch installation
        2. Run the full voice_chatbot.py
        
        **Available now:**
        - ✅ Text chat with AI
        - ✅ Text-to-speech responses
        - ✅ Conversation history
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        💬 Simple Chatbot - Text Only Version<br>
        Run fix_pytorch.py to enable voice features
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
