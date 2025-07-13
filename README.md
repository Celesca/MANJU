# 🎤 Voice Chatbot with ASR & LLM

## Multi-agent AI for Natural Just-in-Time Understanding - Voice Chatbot

A complete voice chatbot solution that combines:
- **ASR (Automatic Speech Recognition)** using Whisper with overlapping chunks
- **LLM (Large Language Model)** using Ollama with Phi3 model
- **TTS (Text-to-Speech)** for audio responses
- **Streamlit** web interface

## 🚀 Features

- **Voice Recording**: Record your voice directly in the web interface
- **File Upload**: Upload audio files for transcription
- **Real-time ASR**: Advanced overlapping chunk processing for better accuracy
- **AI Chat**: Conversation with Phi3 model via Ollama
- **Text-to-Speech**: Hear AI responses spoken back
- **Conversation History**: Track your entire conversation
- **Multiple Input Methods**: Voice, file upload, or text input

## 📋 Prerequisites

1. **Python 3.8+**
2. **Ollama** - Download from [https://ollama.ai](https://ollama.ai)
3. **Audio drivers** for recording (usually built-in on Windows)

## 🛠️ Installation

### Option 1: Automatic Setup (Windows)
```bash
# Run the setup script
setup.bat
```

### Option 2: Manual Setup

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install and setup Ollama:**
```bash
# Download Ollama from https://ollama.ai
# Then install the Phi3 model:
ollama pull phi3
```

3. **Start Ollama server:**
```bash
ollama serve
```

4. **Run the chatbot:**
```bash
streamlit run voice_chatbot.py
```

## 🎯 Usage

1. **Start the application:**
   ```bash
   streamlit run voice_chatbot.py
   ```

2. **Open your browser** and go to the displayed URL (usually `http://localhost:8501`)

3. **Choose your input method:**
   - 🎤 **Voice Recording**: Click "Start Recording" and speak
   - 📁 **File Upload**: Upload an audio file (WAV, MP3, MP4, M4A, FLAC)
   - 💬 **Text Input**: Type your message directly

4. **Interact with the AI:**
   - Your speech gets transcribed using advanced ASR
   - The AI responds using the Phi3 model
   - Responses are spoken back using TTS

## 🔧 Configuration

### ASR Settings
You can modify the ASR configuration in `voice_chatbot.py`:
```python
audio_config = AudioConfig(
    chunk_length_ms=27000,  # 27 seconds per chunk
    overlap_ms=2000,        # 2 seconds overlap
    sample_rate=16000       # 16kHz sample rate
)
```

### LLM Settings
Change the Ollama model:
```python
st.session_state.llm = OllamaLLM(model_name="phi3")  # or "llama2", "mistral", etc.
```

### Recording Settings
Adjust recording duration in the sidebar (3-15 seconds)

## 📁 Project Structure

```
MANJU/
├── voice_chatbot.py      # Main Streamlit application
├── whisper.py           # ASR pipeline with overlapping chunks
├── requirements.txt     # Python dependencies
├── setup.bat           # Windows setup script
└── README.md           # This file
```

## 🎤 ASR Technology

This project uses an advanced **overlapping chunk ASR system**:

- **Chunk Length**: 27 seconds per chunk
- **Overlap**: 2 seconds between chunks
- **Benefits**: Better accuracy for long audio, handles word boundaries
- **Model**: Whisper Thai Large V3 (nectec/Pathumma-whisper-th-large-v3)

## 🧠 LLM Integration

- **Ollama API**: Local LLM inference
- **Phi3 Model**: Microsoft's efficient language model
- **Conversation Memory**: Maintains chat history
- **Flexible**: Easy to switch models

## 🔊 Text-to-Speech

- **Engine**: pyttsx3 (cross-platform)
- **Voice Selection**: Automatic female voice preference
- **Configurable**: Speed and volume adjustable

## 🐛 Troubleshooting

### Common Issues:

1. **"Ollama not connected"**
   - Make sure Ollama is installed and running: `ollama serve`
   - Check if phi3 model is installed: `ollama list`

2. **"Recording not available"**
   - Install audio dependencies: `pip install sounddevice soundfile`
   - Check microphone permissions

3. **"TTS not available"**
   - Install TTS engine: `pip install pyttsx3`
   - On Linux: `sudo apt-get install espeak espeak-data libespeak1`

4. **Slow transcription**
   - GPU acceleration is enabled by default
   - For CPU-only: Set `use_gpu=False` in ProcessingConfig

### Performance Tips:

- **GPU Usage**: Ensure CUDA is available for faster transcription
- **Model Loading**: First transcription may be slower (model loading)
- **Chunk Size**: Reduce chunk_length_ms for faster processing of shorter audio

## 🔒 Privacy

- **Local Processing**: All ASR and LLM processing happens locally
- **No Cloud Calls**: Your audio and conversations stay on your machine
- **Temporary Files**: Audio chunks are automatically cleaned up

## 📊 System Requirements

- **RAM**: 8GB+ recommended (for Whisper model)
- **Storage**: 2GB+ for models
- **CPU**: Multi-core recommended
- **GPU**: CUDA-compatible GPU optional but recommended

## 🤝 Contributing

Feel free to contribute by:
- Adding new TTS voices
- Supporting more LLM models
- Improving the UI
- Adding new audio formats
- Optimizing performance

## 📝 License

This project is open source. Feel free to use and modify as needed.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify Ollama is running and phi3 model is available
4. Check the Streamlit logs for detailed error messages

---

🎉 **Enjoy your voice chatbot!** Speak, chat, and interact naturally with AI!
