# 🎤 Voice Chatbot with ASR & LLM

## Multi-agent AI for Natural Just-in-Time Understanding - Voice Chatbot

A complete voice chatbot solution that combines:
- **ASR (Automatic Speech Recognition)** using Whisper with overlapping chunks
- **LLM (Large Language Model)** using OpenRouter API
- **TTS (Text-to-Speech)** with F5-TTS-THAI and pyttsx3 support
- **Streamlit** web interface

## 🚀 Features

- **Voice Recording**: Record your voice directly in the web interface
- **File Upload**: Upload audio files for transcription
- **Real-time ASR**: Advanced overlapping chunk processing for better accuracy
- **AI Chat**: Conversation with various models via OpenRouter API
- **Advanced TTS**: High-quality Thai text-to-speech with F5-TTS-THAI
- **Multiple TTS Engines**: F5-TTS-THAI for Thai, pyttsx3 for fallback
- **Conversation History**: Track your entire conversation
- **Multiple Input Methods**: Voice, file upload, or text input

## 📋 Prerequisites

1. **Python 3.8+**
2. **OpenRouter API Key** - Get from [https://openrouter.ai](https://openrouter.ai)
3. **Audio drivers** for recording (usually built-in on Windows)
4. **Optional: CUDA** for GPU-accelerated TTS

## 🛠️ Installation

### Option 1: Quick Setup

1. **Install basic dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install F5-TTS-THAI (for better Thai TTS):**
```bash
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git
```
*Or run `install_f5_tts.bat` on Windows*

3. **Set up OpenRouter API key:**
```bash
# Option 1: Environment variable (recommended)
set OPENROUTER_API_KEY=your_api_key_here

# Option 2: Enter in the app sidebar when running
```

4. **Run the chatbot:**
```bash
streamlit run voice_chatbot.py
```

### Option 2: Manual Setup

1. **Install Python dependencies:**
```bash
pip install streamlit transformers torch torchaudio openai
pip install sounddevice soundfile pyttsx3 requests pandas numpy
```

2. **Install F5-TTS-THAI:**
```bash
# For CUDA support (recommended)
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install F5-TTS-THAI
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git
```

## 🎯 Usage

1. **Start the application:**
   ```bash
   streamlit run voice_chatbot.py
   ```

2. **Configure API Key:**
   - Set `OPENROUTER_API_KEY` environment variable, OR
   - Enter API key in the sidebar when app starts

3. **Choose your input method:**
   - 🎤 **Voice Recording**: Click "Start Recording" and speak
   - 📁 **File Upload**: Upload an audio file (WAV, MP3, MP4, M4A, FLAC)
   - 💬 **Text Input**: Type your message directly

4. **Select TTS Engine:**
   - **F5-TTS-THAI**: High-quality neural TTS for Thai (recommended)
   - **pyttsx3**: Traditional TTS engine (fallback)
   - **Auto**: Automatically chooses the best available engine

## 🔧 Configuration

### Available Models (OpenRouter)
- `tencent/hunyuan-a13b-instruct:free` (Default)
- `moonshotai/kimi-k2:free`
- `meta-llama/llama-3.2-11b-vision-instruct:free`
- `microsoft/phi-3-mini-128k-instruct:free`
- And many more free models!

### ASR Settings
Modify in `voice_chatbot.py`:
```python
audio_config = AudioConfig(
    chunk_length_ms=27000,  # 27 seconds per chunk
    overlap_ms=2000,        # 2 seconds overlap
    sample_rate=16000       # 16kHz sample rate
)
```

### TTS Settings
- **F5-TTS-THAI**: Automatic setup with Thai optimization
- **pyttsx3**: Configurable voice, speed, and volume
- **Engine Selection**: Choose in sidebar

## 📁 Project Structure

```
MANJU/
├── voice_chatbot.py        # Main Streamlit application
├── whisper.py             # ASR pipeline with overlapping chunks
├── requirements.txt       # Python dependencies
├── install_f5_tts.bat    # F5-TTS-THAI installation script
└── README.md             # This file
```

## 🎤 ASR Technology

Advanced **overlapping chunk ASR system**:
- **Chunk Length**: 27 seconds per chunk
- **Overlap**: 2 seconds between chunks
- **Benefits**: Better accuracy for long audio, handles word boundaries
- **Model**: Whisper Thai Large V3 (nectec/Pathumma-whisper-th-large-v3)

## 🧠 LLM Integration

- **OpenRouter API**: Access to multiple state-of-the-art models
- **Free Models**: Many free options available
- **Conversation Memory**: Maintains chat history
- **Flexible**: Easy model switching via dropdown

## 🔊 Text-to-Speech

### F5-TTS-THAI (Recommended)
- **High Quality**: Neural speech synthesis for Thai
- **Natural Voice**: More human-like than traditional TTS
- **GPU Accelerated**: CUDA support for faster generation
- **Thai Optimized**: Specifically trained on Thai language data

### pyttsx3 (Fallback)
- **Cross-platform**: Works on Windows, Mac, Linux
- **Voice Selection**: Automatic female voice preference
- **Configurable**: Speed and volume adjustable
- **Reliable**: Always available fallback option

## 🐛 Troubleshooting

### Common Issues:

1. **"OpenRouter not connected"**
   - Set your API key: `set OPENROUTER_API_KEY=your_api_key_here`
   - Or enter it manually in the sidebar
   - Get free API key from [https://openrouter.ai](https://openrouter.ai)

2. **"F5-TTS-THAI initialization failed"**
   - Install dependencies: `pip install torch torchaudio`
   - Install F5-TTS-THAI: `pip install git+https://github.com/VYNCX/F5-TTS-THAI.git`
   - Use pyttsx3 as fallback if F5-TTS fails

3. **"Recording not available"**
   - Install audio dependencies: `pip install sounddevice soundfile`
   - Check microphone permissions

4. **"TTS not available"**
   - Install TTS engine: `pip install pyttsx3`
   - On Linux: `sudo apt-get install espeak espeak-data libespeak1`
   - For F5-TTS: Install CUDA for GPU acceleration

5. **Slow transcription**
   - GPU acceleration is enabled by default
   - For CPU-only: Set `use_gpu=False` in ProcessingConfig

### Performance Tips:

- **GPU Usage**: Ensure CUDA is available for faster F5-TTS generation
- **Model Loading**: First F5-TTS generation may be slower (model loading)
- **API Limits**: Check OpenRouter rate limits if requests fail
- **TTS Engine**: Use "auto" to automatically select best available engine

### F5-TTS-THAI Specific:

- **CUDA Required**: GPU acceleration highly recommended
- **Model Size**: First run downloads ~2GB model
- **Memory**: Requires 4GB+ GPU memory for optimal performance
- **Fallback**: System automatically uses pyttsx3 if F5-TTS fails

## 🔒 Privacy

- **Local ASR**: Speech recognition happens locally with Whisper
- **API Calls**: Only LLM requests sent to OpenRouter (encrypted)
- **No Audio Upload**: Your voice stays on your machine
- **Temporary Files**: Audio chunks automatically cleaned up

## 📊 System Requirements

### Minimum:
- **RAM**: 4GB (for basic operation)
- **Storage**: 3GB+ for models
- **CPU**: Dual-core
- **Network**: Internet for OpenRouter API

### Recommended:
- **RAM**: 8GB+ (for smooth operation)
- **GPU**: CUDA-compatible GPU with 4GB+ VRAM (for F5-TTS)
- **Storage**: 5GB+ for models and cache
- **CPU**: Quad-core or better

## 🤝 Contributing

Feel free to contribute by:
- Adding new TTS voices or engines
- Supporting more OpenRouter models
- Improving the UI/UX
- Adding new audio formats
- Optimizing F5-TTS integration
- Adding multi-language support

## 📝 License

This project is open source. Feel free to use and modify as needed.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify OpenRouter API key is set
4. Check the Streamlit logs for detailed error messages
5. Try using pyttsx3 TTS if F5-TTS fails

## 🔗 Useful Links

- **OpenRouter**: [https://openrouter.ai](https://openrouter.ai) - Get API keys
- **F5-TTS-THAI**: [https://github.com/VYNCX/F5-TTS-THAI](https://github.com/VYNCX/F5-TTS-THAI) - Thai TTS model
- **Whisper Model**: [https://huggingface.co/nectec/Pathumma-whisper-th-large-v3](https://huggingface.co/nectec/Pathumma-whisper-th-large-v3) - Thai ASR model

---

🎉 **Enjoy your enhanced voice chatbot!** Now with OpenRouter LLM and F5-TTS-THAI for premium Thai text-to-speech!
