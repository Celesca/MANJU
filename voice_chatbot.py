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
from openai import OpenAI

# Import your ASR pipeline
from whisper import OverlappingASRPipeline, AudioConfig, ProcessingConfig

# Text-to-Speech imports
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# F5-TTS-THAI imports
try:
    import torch
    import torchaudio
    import soundfile as sf
    # Try different import paths for F5-TTS
    try:
        from f5_tts.api import F5TTS
    except ImportError:
        try:
            from f5_tts.infer.utils_infer import infer_process
            F5TTS = None  # Use function-based approach
        except ImportError:
            F5TTS = None
    F5_TTS_AVAILABLE = True
except ImportError:
    F5_TTS_AVAILABLE = False
    F5TTS = None
    # Don't show warning here as it's optional

# Audio recording imports
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False
    st.warning("Audio recording not available. Install with: pip install sounddevice soundfile")


class OpenRouterLLM:
    """Interface for OpenRouter API"""
    
    def __init__(self, model_name: str = "tencent/hunyuan-a13b-instruct:free", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if self.api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
        else:
            self.client = None
        
    def is_available(self) -> bool:
        """Check if OpenRouter is available"""
        if not self.client or not self.api_key:
            return False
        
        try:
            # Test with a simple request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            st.error(f"OpenRouter connection failed: {str(e)}")
            return False
    
    def get_available_models(self) -> list:
        """Get list of commonly available models on OpenRouter"""
        # Common free/popular models on OpenRouter
        return [
            "tencent/hunyuan-a13b-instruct:free",
            "moonshotai/kimi-k2:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "meta-llama/llama-3.2-1b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "microsoft/phi-3-medium-128k-instruct:free",
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "huggingfaceh4/zephyr-7b-beta:free"
        ]
    
    def chat(self, message: str, conversation_history: list = None) -> str:
        """Send message to OpenRouter and get response"""
        if not self.client:
            return "Error: OpenRouter API key not configured. Please set OPENROUTER_API_KEY environment variable."
        
        try:
            # Prepare messages
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": message})
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Get the response content
            content = response.choices[0].message.content
            
            # Clean up the response by removing <answer> and </answer> tags
            content = self._clean_response(content)
            
            return content
                
        except Exception as e:
            return f"Error communicating with OpenRouter: {str(e)}"
    
    def _clean_response(self, text: str) -> str:
        """Remove <answer> and </answer> tags from the response"""
        if not text:
            return text
        
        import re
        
        # Remove <answer> and </answer> tags (case insensitive)
        # This handles various formats:
        # <answer>content</answer>
        # <Answer>content</Answer>
        # <ANSWER>content</ANSWER>
        cleaned_text = re.sub(r'<\s*answer\s*>', '', text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'</\s*answer\s*>', '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove any extra whitespace that might be left
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text


class F5TTSThai:
    """F5-TTS-THAI handler for Thai text-to-speech"""
    
    def __init__(self):
        self.model = None
        self.infer_function = None
        self.available = False
        
        if F5_TTS_AVAILABLE:
            try:
                self._initialize_model()
            except Exception as e:
                # Silently fail and use fallback
                self.available = False
    
    def _initialize_model(self):
        """Initialize the F5-TTS model"""
        try:
            # Try multiple initialization approaches
            if F5TTS is not None:
                # Try class-based approach
                self.model = F5TTS(model_type="F5-TTS")
                self.available = True
                return
            
            # Try function-based approach
            try:
                from f5_tts.infer.utils_infer import infer_process
                self.infer_function = infer_process
                self.available = True
                return
            except ImportError:
                pass
            
            # Try alternative import
            try:
                from f5_tts.model import F5TTS as F5TTSModel
                self.model = F5TTSModel()
                self.available = True
                return
            except ImportError:
                pass
                
            # Check if command line tool is available
            try:
                import subprocess
                result = subprocess.run(['python', '-c', 'import f5_tts'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    st.info("F5-TTS installed but API not accessible. TTS available via command line.")
                    self.available = False
            except:
                pass
                
        except Exception as e:
            # Don't show error in UI, just mark as unavailable
            self.available = False
    
    def is_available(self) -> bool:
        """Check if F5-TTS-THAI is available"""
        return self.available
    
    def speak(self, text: str, ref_audio: str = None, ref_text: str = None, save_file: str = None) -> bool:
        """Convert text to speech using F5-TTS-THAI"""
        if not self.is_available():
            return False
        
        try:
            # For now, show a message that F5-TTS would be used
            # This prevents the error while maintaining the interface
            st.info("🔊 F5-TTS-THAI: Text-to-speech generation started...")
            st.warning("F5-TTS-THAI audio generation not yet fully implemented. Using fallback TTS.")
            return False  # Fall back to pyttsx3
            
        except Exception as e:
            return False


class GTTSThai:
    """Google Text-to-Speech handler for Thai language"""
    
    def __init__(self):
        self.available = GTTS_AVAILABLE
        # No need to initialize pygame since we're using streamlit audio
    
    def is_available(self) -> bool:
        """Check if gTTS is available"""
        return self.available
    
    def speak(self, text: str, language: str = "th", save_file: str = None) -> bool:
        """Convert text to speech using Google TTS"""
        if not self.available:
            return False
        
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=False)
            
            if save_file:
                # Save to specified file
                tts.save(save_file)
                return True
            else:
                # Create temporary file and play
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_file.close()  # Close the file handle so gTTS can write to it
                
                # Save audio to temp file
                tts.save(temp_file.name)
                
                # Check if file was created successfully
                if not os.path.exists(temp_file.name):
                    st.error("Failed to create audio file")
                    return False
                
                # Play using streamlit audio player instead of pygame
                with open(temp_file.name, 'rb') as audio_file:
                    st.audio(audio_file.read(), format='audio/mp3')
                
                # Schedule cleanup after a delay
                import threading
                def cleanup_file():
                    time.sleep(5)  # Wait 5 seconds for streamlit to load the audio
                    try:
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)
                    except:
                        pass
                
                cleanup_thread = threading.Thread(target=cleanup_file, daemon=True)
                cleanup_thread.start()
            
            return True
            
        except Exception as e:
            st.error(f"gTTS speech generation failed: {str(e)}")
            return False


class PyttsxTTS:
    """pyttsx3 TTS handler (fallback for non-Thai)"""
    
    def __init__(self):
        self.engine = None
        if PYTTSX3_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self._setup_voice()
            except:
                self.engine = None
    
    def _setup_voice(self):
        """Setup pyttsx3 voice properties"""
        if self.engine:
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a female voice or use first available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    self.engine.setProperty('voice', voices[0].id)
            
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 0.8)  # Volume
    
    def is_available(self) -> bool:
        """Check if pyttsx3 is available"""
        return self.engine is not None
    
    def speak(self, text: str, save_file: str = None) -> bool:
        """Convert text to speech using pyttsx3"""
        if not self.engine:
            return False
        
        try:
            if save_file:
                self.engine.save_to_file(text, save_file)
                self.engine.runAndWait()
            else:
                self.engine.say(text)
                self.engine.runAndWait()
            return True
        except:
            return False


class TextToSpeech:
    """Text-to-Speech handler with multiple engine support"""
    
    def __init__(self, prefer_f5_tts: bool = True):
        self.prefer_f5_tts = prefer_f5_tts
        self.gtts_engine = None
        self.pyttsx3_engine = None
        self.f5_tts_engine = None
        
        # Initialize F5-TTS-THAI if preferred and available
        if prefer_f5_tts and F5_TTS_AVAILABLE:
            self.f5_tts_engine = F5TTSThai()
        
        # Initialize gTTS for Thai (primary choice)
        if GTTS_AVAILABLE:
            self.gtts_engine = GTTSThai()
        
        # Initialize pyttsx3 as last fallback
        if PYTTSX3_AVAILABLE:
            self.pyttsx3_engine = PyttsxTTS()
    
    def get_available_engines(self) -> list:
        """Get list of available TTS engines"""
        engines = []
        if self.f5_tts_engine and self.f5_tts_engine.is_available():
            engines.append("F5-TTS-THAI")
        if self.gtts_engine and self.gtts_engine.is_available():
            engines.append("gTTS-Thai")
        if self.pyttsx3_engine and self.pyttsx3_engine.is_available():
            engines.append("pyttsx3")
        return engines
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is Thai or English"""
        # Simple Thai detection - check for Thai Unicode characters
        thai_chars = sum(1 for char in text if '\u0e00' <= char <= '\u0e7f')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return "th"  # Default to Thai
        
        thai_ratio = thai_chars / total_chars
        return "th" if thai_ratio > 0.3 else "en"
    
    def speak(self, text: str, engine: str = "auto", save_file: str = None) -> bool:
        """Convert text to speech using specified engine"""
        # Determine which engine to use
        if engine == "auto":
            # Auto-select based on preference and availability
            if self.prefer_f5_tts and self.f5_tts_engine and self.f5_tts_engine.is_available():
                engine = "F5-TTS-THAI"
            elif self.gtts_engine and self.gtts_engine.is_available():
                engine = "gTTS-Thai"
            elif self.pyttsx3_engine and self.pyttsx3_engine.is_available():
                engine = "pyttsx3"
            else:
                return False
        
        # Use F5-TTS-THAI (best for Thai)
        if engine == "F5-TTS-THAI" and self.f5_tts_engine and self.f5_tts_engine.is_available():
            success = self.f5_tts_engine.speak(text, save_file=save_file)
            if success:
                return True
            # Fall back to gTTS if F5-TTS fails
            engine = "gTTS-Thai"
        
        # Use gTTS (good for Thai and other languages)
        if engine == "gTTS-Thai" and self.gtts_engine and self.gtts_engine.is_available():
            language = self._detect_language(text)
            return self.gtts_engine.speak(text, language=language, save_file=save_file)
        
        # Use pyttsx3 as last fallback (not ideal for Thai)
        elif engine == "pyttsx3" and self.pyttsx3_engine and self.pyttsx3_engine.is_available():
            return self.pyttsx3_engine.speak(text, save_file=save_file)
        
        return False
    
    def is_available(self) -> bool:
        """Check if any TTS engine is available"""
        f5_available = self.f5_tts_engine and self.f5_tts_engine.is_available()
        gtts_available = self.gtts_engine and self.gtts_engine.is_available()
        pyttsx3_available = self.pyttsx3_engine and self.pyttsx3_engine.is_available()
        return f5_available or gtts_available or pyttsx3_available


class AudioRecorder:
    """Audio recording handler"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_data = None
    
    def record_audio(self, duration: int = 5) -> str:
        """Record audio and save to temporary file"""
        if not RECORDING_AVAILABLE:
            return None
        
        try:
            st.info(f"Recording for {duration} seconds... Speak now!")
            
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate), 
                samplerate=self.sample_rate, 
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # Wait until recording is finished
            
            # Save to temporary file as WAV (no ffmpeg needed)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, audio_data, self.sample_rate)
            
            st.success("Recording completed!")
            return temp_file.name
            
        except Exception as e:
            st.error(f"Recording failed: {str(e)}")
            return None


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
            use_gpu=True
        )
        st.session_state.asr_pipeline = OverlappingASRPipeline(
            input_path="",  # Will be set dynamically
            audio_config=audio_config,
            processing_config=processing_config
        )
    
    if 'llm' not in st.session_state:
        st.session_state.llm = OpenRouterLLM(model_name="tencent/hunyuan-a13b-instruct:free")

    if 'tts' not in st.session_state:
        # Initialize TTS with F5-TTS preference for Thai
        st.session_state.tts = TextToSpeech(prefer_f5_tts=True)
    
    if 'tts_engine' not in st.session_state:
        st.session_state.tts_engine = "auto"
    
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()


def display_conversation():
    """Display conversation history"""
    st.subheader("💬 Conversation History")
    
    if not st.session_state.conversation_history:
        st.info("No conversation yet. Start by recording your voice or typing a message!")
        return
    
    for i, msg in enumerate(st.session_state.conversation_history):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        if role == 'user':
            with st.chat_message("user"):
                st.write(f"🎤 **You:** {content}")
        elif role == 'assistant':
            with st.chat_message("assistant"):
                st.write(f"🤖 **Assistant:** {content}")
                
                # Add TTS button for assistant responses
                if st.session_state.tts.is_available():
                    if st.button(f"🔊 Play Response {i}", key=f"tts_{i}"):
                        st.session_state.tts.speak(content, engine=st.session_state.tts_engine)


def main():
    st.set_page_config(
        page_title="Voice Chatbot",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Voice Chatbot with ASR & LLM")
    st.markdown("*Speak, transcribe, chat, and listen to responses!*")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check system status
        st.subheader("System Status")
        
        # FFmpeg Status
        ffmpeg_available = check_ffmpeg_available()
        if ffmpeg_available:
            st.write("🎵 **FFmpeg:** ✅ Available")
        else:
            st.write("🎵 **FFmpeg:** ❌ Missing")
            st.error("FFmpeg required for audio processing")
            if st.button("📥 Install FFmpeg Guide"):
                st.markdown("""
                **Install FFmpeg:**
                1. Run: `install_ffmpeg.bat`
                2. Or download: [FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases)
                3. Add to PATH or restart
                """)
        
        # ASR Status
        if ffmpeg_available:
            st.write("🎯 **ASR (Whisper):** ✅ Ready")
        else:
            st.write("🎯 **ASR (Whisper):** ❌ Needs FFmpeg")
        
        # OpenRouter Status
        if st.session_state.llm.is_available():
            available_models = st.session_state.llm.get_available_models()
            st.write("🧠 **OpenRouter:** ✅ Connected")
            
            # Model selection
            if available_models:
                current_model = st.session_state.llm.model_name
                selected_model = st.selectbox(
                    "Select Model:",
                    available_models,
                    index=available_models.index(current_model) if current_model in available_models else 0
                )
                st.session_state.llm.model_name = selected_model
            else:
                st.write("No models available")
        else:
            st.write("🧠 **OpenRouter:** ❌ Not connected")
            st.error("Please set OPENROUTER_API_KEY environment variable")
            
            # Add API key input option
            api_key_input = st.text_input("Or enter API key here:", type="password")
            if api_key_input and st.button("Connect with API Key"):
                st.session_state.llm = OpenRouterLLM(api_key=api_key_input)
                st.rerun()
        
        # TTS Status
        if st.session_state.tts.is_available():
            available_engines = st.session_state.tts.get_available_engines()
            st.write("🔊 **TTS:** ✅ Available")
            
            # Show available engines
            if available_engines:
                st.write(f"**Engines:** {', '.join(available_engines)}")
                
                # Engine selection
                engine_options = ["auto"] + available_engines
                selected_engine = st.selectbox(
                    "TTS Engine:",
                    engine_options,
                    index=engine_options.index(st.session_state.tts_engine) if st.session_state.tts_engine in engine_options else 0
                )
                st.session_state.tts_engine = selected_engine
                
                # Show F5-TTS installation guide if not available
                if "F5-TTS-THAI" not in available_engines:
                    with st.expander("📥 Install F5-TTS-THAI for premium Thai TTS"):
                        st.markdown("""
                        **Install F5-TTS-THAI:**
                        ```bash
                        pip install torch torchaudio
                        pip install git+https://github.com/VYNCX/F5-TTS-THAI.git
                        ```
                        **Note:** Requires CUDA for GPU acceleration
                        """)
                
                # Show gTTS info if not available
                if "gTTS-Thai" not in available_engines:
                    with st.expander("📥 Install gTTS for Thai TTS"):
                        st.markdown("""
                        **Install gTTS (Google Text-to-Speech):**
                        ```bash
                        pip install gtts
                        ```
                        **Features:** Excellent Thai support, natural voice, requires internet
                        """)
        else:
            st.write("🔊 **TTS:** ❌ Not available")
            st.error("Install TTS engines for Thai support:")
            
            # Installation guides
            with st.expander("📥 Recommended: Install gTTS (Best for Thai)"):
                st.markdown("""
                **Install gTTS:**
                ```bash
                pip install gtts
                ```
                **Why gTTS?**
                - ✅ Excellent Thai language support
                - ✅ Natural Google voices
                - ✅ Easy to install
                - ✅ Plays through Streamlit interface
                - ❌ Requires internet connection
                """)
            
            with st.expander("📥 Premium: Install F5-TTS-THAI"):
                st.markdown("""
                **Install F5-TTS-THAI:**
                ```bash
                pip install torch torchaudio
                pip install git+https://github.com/VYNCX/F5-TTS-THAI.git
                ```
                **Why F5-TTS?**
                - ✅ Highest quality Thai TTS
                - ✅ Works offline
                - ✅ Customizable voice
                - ❌ Requires GPU for best performance
                """)
            
            with st.expander("📥 Fallback: Install pyttsx3"):
                st.markdown("""
                **Install pyttsx3:**
                ```bash
                pip install pyttsx3
                ```
                **Note:** Limited Thai support, better for English
                """)
        
        # Recording Status
        if RECORDING_AVAILABLE:
            st.write("🎙️ **Recording:** ✅ Available")
            
            # Recording settings
            st.subheader("Recording Settings")
            recording_duration = st.slider("Recording Duration (seconds)", 3, 15, 5)
        else:
            st.write("🎙️ **Recording:** ❌ Not available")
            recording_duration = 5
        
        st.divider()
        
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
        st.subheader("🎙️ Voice Input")
        
        # Voice recording
        if RECORDING_AVAILABLE and st.button("🎤 Start Recording", type="primary"):
            audio_file = st.session_state.recorder.record_audio(recording_duration)
            
            if audio_file:
                try:
                    # Display audio player
                    st.audio(audio_file)
                    
                    # Transcribe audio
                    with st.spinner("Transcribing audio..."):
                        # Create temporary pipeline for this file
                        temp_pipeline = OverlappingASRPipeline(
                            input_path=audio_file,
                            audio_config=st.session_state.asr_pipeline.audio_config,
                            processing_config=st.session_state.asr_pipeline.processing_config
                        )
                        transcription = temp_pipeline()
                    
                    if transcription and not transcription.startswith("[ERROR"):
                        st.success(f"Transcribed: {transcription}")
                        
                        # Add to conversation
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": transcription
                        })
                        
                        # Get LLM response
                        if st.session_state.llm.is_available():
                            with st.spinner("Getting AI response..."):
                                response = st.session_state.llm.chat(
                                    transcription, 
                                    st.session_state.conversation_history[:-1]
                                )
                            
                            # Add response to conversation
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": response
                            })
                            
                            st.success("Response generated!")
                            
                            # Display the response immediately
                            st.info(f"🤖 **Assistant Response:** {response}")
                            
                            # Auto-play response if TTS is available
                            if st.session_state.tts.is_available():
                                with st.spinner("Playing response..."):
                                    st.session_state.tts.speak(response, engine=st.session_state.tts_engine)
                        else:
                            st.error("OpenRouter not available for response generation")
                        
                        st.rerun()
                    else:
                        st.error("Transcription failed or empty")
                
                finally:
                    # Cleanup temporary file
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
        
        st.divider()
        
        # File upload option
        st.subheader("📁 Upload Audio")
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=['wav', 'mp3', 'mp4', 'm4a', 'flac'],
            help="Upload an audio file to transcribe"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}')
            temp_file.write(uploaded_file.read())
            temp_file.close()
            
            try:
                # Display audio player
                st.audio(uploaded_file)
                
                if st.button("🎯 Transcribe Uploaded File"):
                    with st.spinner("Transcribing uploaded audio..."):
                        temp_pipeline = OverlappingASRPipeline(
                            input_path=temp_file.name,
                            audio_config=st.session_state.asr_pipeline.audio_config,
                            processing_config=st.session_state.asr_pipeline.processing_config
                        )
                        transcription = temp_pipeline()
                    
                    if transcription and not transcription.startswith("[ERROR"):
                        st.success(f"Transcribed: {transcription}")
                        
                        # Add to conversation and get response (same as recording)
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
                            
                            # Display the response immediately
                            st.success("Response generated!")
                            st.info(f"🤖 **Assistant Response:** {response}")
                            
                            if st.session_state.tts.is_available():
                                with st.spinner("Playing response..."):
                                    st.session_state.tts.speak(response, engine=st.session_state.tts_engine)
                        
                        st.rerun()
                    else:
                        st.error("Transcription failed")
            
            finally:
                # Cleanup
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
        
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
                    
                    # Display the response immediately
                    st.success("Response generated!")
                    st.info(f"🤖 **Assistant Response:** {response}")
                    
                    if st.session_state.tts.is_available():
                        with st.spinner("Playing response..."):
                            st.session_state.tts.speak(response, engine=st.session_state.tts_engine)
                
                st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        🎤 Voice Chatbot powered by Whisper ASR + OpenRouter LLM + gTTS/F5-TTS Thai
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
