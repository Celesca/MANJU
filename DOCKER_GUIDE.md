# MANJU Voice Chatbot - Quick Start Guide

## 🚀 Quick Start with Docker (Recommended)

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- 8GB+ RAM recommended
- 5GB+ free disk space

### Option 1: Automated Setup (Windows)
```cmd
docker-setup.bat
```

### Option 2: Manual Setup (All Platforms)
```bash
# Build and start all services
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Access the Application
- **Voice Chatbot**: http://localhost:8501
- **Ollama API**: http://localhost:11434

---

## 🖥️ Local Installation (Windows)

### Automated Setup
```cmd
complete_setup.bat
```

### Manual Setup
1. Install Python 3.11+
2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
3. Install ffmpeg:
   ```cmd
   install_ffmpeg.bat
   ```
4. Run the application:
   ```cmd
   streamlit run voice_chatbot.py
   ```

---

## 🏗️ Architecture

```
MANJU Voice Chatbot System
├── 🎤 Voice Input (Streamlit UI)
├── 🧠 ASR (Whisper via Transformers)
├── 🤖 LLM (Phi3 via Ollama)
├── 🔊 TTS (pyttsx3 - local only)
└── 📦 Docker Deployment
```

---

## 📁 Project Structure

```
MANJU/
├── 🐳 Docker Files
│   ├── docker-compose.yml       # Multi-service orchestration
│   ├── Dockerfile              # Python app container
│   ├── docker-setup.bat        # Windows setup script
│   ├── docker-setup.sh         # Linux/Mac setup script
│   └── .dockerignore           # Docker build exclusions
│
├── 🎯 Core Application
│   ├── voice_chatbot.py        # Main Streamlit app (local)
│   ├── voice_chatbot_docker.py # Docker-optimized app
│   ├── whisper.py              # OOP ASR pipeline
│   └── simple_chatbot.py       # Basic CLI version
│
├── ⚙️ Setup & Dependencies
│   ├── requirements.txt        # Local Python dependencies
│   ├── requirements-docker.txt # Docker Python dependencies
│   ├── setup.py               # Package configuration
│   └── complete_setup.bat     # Comprehensive Windows setup
│
├── 🔧 Utilities & Fixes
│   ├── fix_all_issues.py      # Dependency troubleshooting
│   ├── fix_pytorch.py         # PyTorch-specific fixes
│   ├── install_ffmpeg.bat     # FFmpeg installation
│   └── test_asr.py            # ASR testing utility
│
└── 📚 Documentation
    ├── README.md              # This file
    ├── FIX_INSTRUCTIONS.md    # Troubleshooting guide
    └── DOCKER_GUIDE.md        # Docker deployment details
```

---

## 🎛️ Features

### 🎤 Automatic Speech Recognition (ASR)
- **Model**: OpenAI Whisper (via Transformers)
- **Backend**: PyTorch (CPU optimized)
- **Features**: Overlapping chunking, noise handling, file upload support
- **Formats**: WAV, MP3, M4A, FLAC, OGG

### 🧠 Large Language Model (LLM)
- **Model**: Microsoft Phi3-mini
- **Backend**: Ollama
- **Features**: Context-aware responses, conversational memory
- **API**: RESTful interface at localhost:11434

### 🔊 Text-to-Speech (TTS)
- **Engine**: pyttsx3 (local only)
- **Features**: Configurable voice, speed, volume
- **Note**: Disabled in Docker (GUI limitations)

### 🌐 Web Interface
- **Framework**: Streamlit
- **Features**: Real-time recording, file upload, chat history
- **Access**: http://localhost:8501

---

## 🐳 Docker Services

### 1. Ollama Service
- **Purpose**: LLM inference server
- **Model**: Phi3-mini (4GB)
- **Port**: 11434
- **Features**: GPU support, health checks

### 2. Model Downloader
- **Purpose**: Download Phi3 model on first run
- **Type**: Init container
- **Features**: Automatic model verification

### 3. Voice Chatbot
- **Purpose**: Streamlit web application
- **Port**: 8501
- **Features**: ASR, LLM integration, file upload

---

## ⚙️ Configuration

### Environment Variables
```bash
# Ollama Configuration
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=phi3

# Application Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Docker Compose Override
Create `docker-compose.override.yml` for customization:
```yaml
version: '3.8'
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 🔧 Troubleshooting

### Docker Issues
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f ollama
docker-compose logs -f chatbot

# Restart services
docker-compose restart

# Full reset
docker-compose down -v
docker-compose up --build
```

### Local Issues
```cmd
# Fix all dependencies
fix_all_issues.py

# Fix PyTorch specifically
fix_pytorch.py

# Reinstall ffmpeg
install_ffmpeg.bat
```

### Common Problems

1. **Ollama not responding**
   - Wait for model download (5-10 minutes)
   - Check: `docker-compose logs ollama`

2. **Audio not working**
   - Ensure ffmpeg is installed
   - Check file format support

3. **Memory issues**
   - Increase Docker memory limit (8GB+)
   - Close other applications

4. **Port conflicts**
   - Change ports in docker-compose.yml
   - Kill processes using ports 8501/11434

---

## 🚀 Performance Tips

### Hardware Requirements
- **Minimum**: 4GB RAM, 2GB disk
- **Recommended**: 8GB+ RAM, 5GB+ disk
- **GPU**: Optional (Nvidia with CUDA)

### Optimization
- Use CPU-only PyTorch for stability
- Enable GPU in Docker for speed
- Adjust chunk size for memory usage
- Use SSD for model storage

---

## 🔐 Security Notes

- Services run on localhost only
- No authentication required (development)
- Audio files stored temporarily
- Models downloaded over HTTPS

---

## 📞 Support

### Getting Help
1. Check logs: `docker-compose logs`
2. Review FIX_INSTRUCTIONS.md
3. Run diagnostic scripts
4. Check Docker/Python versions

### Known Limitations
- TTS not available in Docker
- Recording not available in Docker
- Windows requires special setup
- GPU support needs configuration

---

## 🎯 Usage Examples

### 1. Voice Conversation
1. Open http://localhost:8501
2. Click "🎤 Start Recording"
3. Speak your question
4. Wait for ASR → LLM → Response

### 2. File Upload
1. Upload audio file (WAV/MP3/etc.)
2. Wait for transcription
3. Get LLM response
4. Download TTS audio (local only)

### 3. API Usage
```python
import requests

# Transcribe audio
response = requests.post("http://localhost:8501/api/transcribe", 
                        files={"audio": open("audio.wav", "rb")})

# Chat with LLM
response = requests.post("http://localhost:11434/api/generate", 
                        json={"model": "phi3", "prompt": "Hello!"})
```

---

## 🎉 Conclusion

MANJU Voice Chatbot provides a complete voice-to-voice AI experience with:
- ✅ Robust ASR with Whisper
- ✅ Intelligent responses with Phi3
- ✅ Easy deployment with Docker
- ✅ Cross-platform compatibility
- ✅ Professional-grade error handling

Ready to chat with AI using your voice! 🎤🤖

---

*Last updated: December 2024*
