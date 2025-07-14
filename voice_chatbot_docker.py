import streamlit as st
import tempfile
import os
import requests
import json
from pathlib import Path
import time
import base64
from io import BytesIO
import pandas as pd

# Import your ASR pipeline
from whisper import OverlappingASRPipeline, AudioConfig, ProcessingConfig

# Text-to-Speech imports - Intentionally disabled in Docker
TTS_AVAILABLE = False  # Disabled in Docker environment

# Audio recording imports - Disabled in Docker
RECORDING_AVAILABLE = False
st.info("🐳 Running in Docker mode - File upload only (no browser recording)")


class OllamaLLM:
    """Interface for Ollama API"""
    
    def __init__(self, model_name: str = "phi3", base_url: str = None):
        self.model_name = model_name
        # Use environment variable or default
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.chat_url = f"{self.base_url}/api/chat"
        
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except:
            pass
        return []
    
    def chat(self, message: str, conversation_history: list = None) -> str:
        """Send message to Ollama and get response"""
        try:
            # Prepare messages
            messages = conversation_history or []
            messages.append({"role": "user", "content": message})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False
            }
            
            response = requests.post(
                self.chat_url, 
                json=payload, 
                timeout=60
            )
            
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
    """Text-to-Speech handler - Disabled in Docker"""
    
    def __init__(self):
        self.engine = None
        # TTS disabled in Docker environment
        
    def speak(self, text: str) -> bool:
        """Convert text to speech - Disabled in Docker"""
        st.info("🐳 TTS disabled in Docker mode")
        return False
    
    def is_available(self) -> bool:
        """Check if TTS is available"""
        return False


def check_ffmpeg_available():
    """Check if FFmpeg is available"""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'asr_pipeline' not in st.session_state:
        # Initialize ASR pipeline
        audio_config = AudioConfig(
            chunk_length_ms=27000,
            overlap_ms=2000,
            sample_rate=16000
        )
        processing_config = ProcessingConfig(
            model_name="nectec/Pathumma-whisper-th-large-v3",
            batch_size=2,
            use_gpu=False  # CPU only in Docker
        )
        st.session_state.asr_pipeline = OverlappingASRPipeline(
            input_path="",  # Will be set dynamically
            audio_config=audio_config,
            processing_config=processing_config
        )
    
    if 'llm' not in st.session_state:
        st.session_state.llm = OllamaLLM(model_name="phi3")
    
    if 'tts' not in st.session_state:
        st.session_state.tts = TextToSpeech()


def display_conversation():
    """Display conversation history"""
    st.subheader("💬 Conversation History")
    
    if not st.session_state.conversation_history:
        st.info("No conversation yet. Upload an audio file or type a message!")
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


def main():
    st.set_page_config(
        page_title="Voice Chatbot (Docker)",
        page_icon="🐳",
        layout="wide"
    )
    
    st.title("🐳 Voice Chatbot - Docker Edition")
    st.markdown("*Upload audio files for transcription and chat with AI*")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check system status
        st.subheader("System Status")
        
        # Docker info
        st.write("🐳 **Environment:** Docker Container")
        
        # FFmpeg Status
        ffmpeg_available = check_ffmpeg_available()
        if ffmpeg_available:
            st.write("🎵 **FFmpeg:** ✅ Available")
        else:
            st.write("🎵 **FFmpeg:** ❌ Missing")
        
        # ASR Status
        if ffmpeg_available:
            st.write("🎯 **ASR (Whisper):** ✅ Ready")
        else:
            st.write("🎯 **ASR (Whisper):** ❌ Needs FFmpeg")
        
        # Ollama Status
        ollama_status = "Checking..."
        st.write(f"🧠 **Ollama:** {ollama_status}")
        
        if st.session_state.llm.is_available():
            available_models = st.session_state.llm.get_available_models()
            st.write("🧠 **Ollama:** ✅ Connected")
            
            # Model selection
            if available_models:
                selected_model = st.selectbox(
                    "Select Model:",
                    available_models,
                    index=available_models.index("phi3") if "phi3" in available_models else 0
                )
                st.session_state.llm.model_name = selected_model
            else:
                st.write("No models available - downloading phi3...")
        else:
            st.write("🧠 **Ollama:** ❌ Not connected")
            st.error("Waiting for Ollama service to start...")
        
        # TTS Status
        st.write("🔊 **TTS:** ❌ Disabled (Docker)")
        
        # Recording Status
        st.write("🎙️ **Recording:** ❌ Disabled (Docker)")
        
        st.divider()
        
        # Docker info
        st.subheader("🐳 Docker Info")
        st.markdown("""
        **Features in Docker:**
        - ✅ Audio file upload
        - ✅ ASR transcription  
        - ✅ AI chat with Ollama
        - ✅ Conversation history
        - ❌ Browser recording
        - ❌ Text-to-speech
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
        st.subheader("📁 Audio Upload")
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg'],
            help="Upload an audio file to transcribe"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            upload_dir = "/app/audio_uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            temp_path = os.path.join(upload_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            try:
                # Display audio player
                st.audio(uploaded_file)
                
                if st.button("🎯 Transcribe Audio File", type="primary"):
                    if not ffmpeg_available:
                        st.error("FFmpeg not available. Please check Docker setup.")
                        return
                    
                    with st.spinner("Transcribing audio..."):
                        temp_pipeline = OverlappingASRPipeline(
                            input_path=temp_path,
                            audio_config=st.session_state.asr_pipeline.audio_config,
                            processing_config=st.session_state.asr_pipeline.processing_config
                        )
                        transcription = temp_pipeline()
                    
                    if transcription and not transcription.startswith("[ERROR"):
                        st.success(f"Transcribed: {transcription}")
                        
                        # Add to conversation and get response
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": transcription
                        })
                        
                        if st.session_state.llm.is_available():
                            with st.spinner("Getting AI response..."):
                                response = st.session_state.llm.chat(
                                    transcription, 
                                    st.session_state.conversation_history[:-1]
                                )
                            
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": response
                            })
                            
                            st.success("Response generated!")
                        else:
                            st.error("Ollama not available for response generation")
                        
                        st.rerun()
                    else:
                        st.error("Transcription failed")
            
            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        st.divider()
        
        # Text input fallback
        st.subheader("💬 Text Input")
        text_input = st.text_area("Type your message:", height=100)
        
        if st.button("💬 Send Text Message"):
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
                else:
                    st.error("Ollama not available")
                
                st.rerun()
        
        st.divider()
        
        # Export conversation
        st.subheader("💾 Export")
        if st.session_state.conversation_history:
            if st.button("📄 Download Conversation"):
                # Create CSV export
                df = pd.DataFrame(st.session_state.conversation_history)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"conversation_{int(time.time())}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        🐳 Voice Chatbot Docker Edition<br>
        Upload audio files • AI Chat • Export conversations
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
