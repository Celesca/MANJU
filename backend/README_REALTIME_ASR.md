# Thai ASR Real-time WebSocket Server

## Overview
This real-time Thai ASR WebSocket server provides high-performance speech recognition using optimized faster-whisper models with 80% GPU utilization. It supports real-time audio streaming and transcription via WebSocket connections.

## Features

### 🚀 Performance Optimizations
- **80% GPU utilization** for maximum hardware efficiency
- **faster-whisper backend** for 2-4x speed improvement
- **Real-time processing** with low latency
- **Optimized audio chunking** and buffering
- **Thai language specific** models and settings

### 🎤 Audio Processing
- **Voice Activity Detection (VAD)** for speech detection
- **Noise reduction** filtering
- **Audio resampling** to 16kHz
- **Real-time chunking** with overlap processing
- **Multiple audio format** support

### 🌐 WebSocket Interface
- **Dual WebSocket design** (control + audio)
- **Multi-client support** with broadcasting
- **Real-time transcription** updates
- **Final transcription** with metadata
- **Model switching** on-the-fly

## Architecture

### WebSocket Endpoints

1. **Control WebSocket** (`ws://localhost:8765`)
   - Model management (load, list)
   - Recording control (start, stop)
   - Server status and configuration
   - Error handling and responses

2. **Audio WebSocket** (`ws://localhost:8766`)
   - Audio data streaming
   - Real-time transcription results
   - Final transcription with metadata
   - Recording state updates

### Audio Processing Pipeline

```
Microphone → Audio Chunks → VAD → Noise Reduction → ASR Model → Transcription
                ↓              ↓           ↓            ↓           ↓
            WebSocket    Buffer/Queue  GPU Processing  Broadcasting  Clients
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_realtime.txt

# For GPU support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Start Server

**Windows:**
```cmd
start_realtime_asr.bat
```

**Manual:**
```bash
python realtime_thai_asr_server.py
```

### 3. Test with Web Client

1. Start the server
2. Open `realtime_thai_asr_client.html` in a web browser
3. Click "Connect" to connect to the server
4. Select a Thai model (recommended: `biodatlab-medium-faster`)
5. Click "Start Recording" and speak in Thai

### 4. Test with Python Client

```bash
python realtime_thai_asr_client.py
```

## API Reference

### Control WebSocket Messages

#### Connect
```json
{
  "type": "connected",
  "server": "Thai ASR Real-time Server",
  "version": "1.0.0",
  "current_model": "biodatlab-medium-faster"
}
```

#### Load Model
```json
// Request
{
  "type": "load_model",
  "model_id": "biodatlab-medium-faster"
}

// Response
{
  "type": "model_loaded",
  "model_id": "biodatlab-medium-faster",
  "status": "success"
}
```

#### Get Available Models
```json
// Request
{
  "type": "get_models"
}

// Response
{
  "type": "available_models",
  "models": [
    {
      "id": "biodatlab-medium-faster",
      "name": "Biodatlab Whisper Thai Medium (Faster)",
      "type": "faster_whisper",
      "language": "th",
      "description": "Thai-optimized medium model...",
      "performance_tier": "balanced",
      "recommended": true
    }
  ]
}
```

#### Start/Stop Recording
```json
// Start Recording
{
  "type": "start_recording"
}

// Stop Recording
{
  "type": "stop_recording"
}
```

### Audio WebSocket Messages

#### Audio Chunk
```json
{
  "type": "audio_chunk",
  "data": "base64_encoded_audio_data"
}
```

#### Audio End
```json
{
  "type": "audio_end"
}
```

#### Real-time Transcription
```json
{
  "type": "realtime_transcription",
  "text": "สวัสดีครับ",
  "timestamp": "2025-09-03T10:30:00.000Z",
  "is_final": false
}
```

#### Final Transcription
```json
{
  "type": "final_transcription",
  "text": "สวัสดีครับ ผมชื่อโจน",
  "duration": 2.5,
  "processing_time": 0.8,
  "speed_ratio": 3.1,
  "model": "Vinxscribe/biodatlab-whisper-th-medium-faster",
  "timestamp": "2025-09-03T10:30:02.500Z",
  "is_final": true
}
```

## Available Models

### Recommended Thai Models

1. **biodatlab-medium-faster** ⭐
   - **Best balance** of speed and accuracy
   - Thai language optimized
   - GPU efficient processing
   - Recommended for most use cases

2. **biodatlab-faster** ⭐
   - **Fastest processing** speed
   - Good accuracy for Thai
   - Lowest latency
   - Ideal for real-time applications

3. **pathumma-large** ⭐
   - **Highest accuracy** for Thai
   - NECTEC optimized model
   - Slower but most precise
   - Best for critical applications

### Performance Comparison

| Model | Speed | Accuracy | GPU Memory | Use Case |
|-------|-------|----------|------------|----------|
| biodatlab-faster | 🚀🚀🚀 | ⭐⭐⭐ | Low | Real-time chat |
| biodatlab-medium-faster | 🚀🚀 | ⭐⭐⭐⭐ | Medium | General purpose |
| pathumma-large | 🚀 | ⭐⭐⭐⭐⭐ | High | Critical accuracy |

## Configuration

### Audio Settings

```python
# Default audio configuration
SAMPLE_RATE = 16000      # 16kHz for optimal ASR
CHUNK_SIZE = 1024        # Audio chunk size
CHANNELS = 1             # Mono audio
AUDIO_FORMAT = paInt16   # 16-bit PCM
```

### GPU Optimization

```python
# GPU configuration (applied automatically)
gpu_memory_fraction = 0.8    # 80% GPU utilization
compute_type = "float16"     # GPU optimized precision
num_workers = 4              # Parallel processing
batch_size = 8               # Increased for GPU efficiency
```

### VAD (Voice Activity Detection)

```python
# VAD settings
vad_threshold = 0.02         # Energy threshold for speech
silence_threshold = 30       # Chunks of silence before stop
min_speech_chunks = 20       # Minimum chunks for valid speech
```

## Integration Examples

### JavaScript/Web Integration

```javascript
// Connect to WebSocket
const audioWs = new WebSocket('ws://localhost:8766');

// Send audio data
function sendAudioChunk(audioData) {
    const base64Audio = btoa(String.fromCharCode.apply(null, audioData));
    audioWs.send(JSON.stringify({
        type: 'audio_chunk',
        data: base64Audio
    }));
}

// Handle transcription results
audioWs.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'realtime_transcription') {
        updateRealtimeDisplay(data.text);
    } else if (data.type === 'final_transcription') {
        updateFinalDisplay(data.text, data);
    }
};
```

### Python Integration

```python
import asyncio
import websockets
import json
import base64

async def connect_to_asr():
    uri = "ws://localhost:8766"
    async with websockets.connect(uri) as websocket:
        # Send audio chunk
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        await websocket.send(json.dumps({
            'type': 'audio_chunk',
            'data': audio_b64
        }))
        
        # Receive transcription
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Transcription: {data['text']}")
```

### Node.js Integration

```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8766');

ws.on('open', () => {
    console.log('Connected to Thai ASR server');
});

ws.on('message', (data) => {
    const message = JSON.parse(data);
    if (message.type === 'final_transcription') {
        console.log('Thai text:', message.text);
    }
});

// Send audio data
function sendAudio(audioBuffer) {
    const base64Audio = audioBuffer.toString('base64');
    ws.send(JSON.stringify({
        type: 'audio_chunk',
        data: base64Audio
    }));
}
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```
   Error: WebSocket connection failed
   ```
   - **Solution**: Make sure the server is running on ports 8765 and 8766
   - Check firewall settings
   - Verify server startup logs

2. **No Audio Detected**
   ```
   Warning: No speech detected in audio stream
   ```
   - **Solution**: Check microphone permissions
   - Verify audio device is working
   - Adjust VAD threshold settings

3. **GPU Not Used**
   ```
   Info: Using CPU (consider GPU for better performance)
   ```
   - **Solution**: Install CUDA toolkit
   - Install GPU-enabled PyTorch
   - Check GPU availability with `nvidia-smi`

4. **Model Loading Failed**
   ```
   Error: Failed to load model biodatlab-medium-faster
   ```
   - **Solution**: Check internet connection for model download
   - Verify model ID is correct
   - Clear Hugging Face cache if needed

5. **High Latency**
   ```
   Warning: Processing time > 1.0s per chunk
   ```
   - **Solution**: Reduce audio chunk size
   - Enable GPU acceleration
   - Use faster model (biodatlab-faster)

### Performance Optimization

1. **For Low Latency**:
   - Use `biodatlab-faster` model
   - Reduce chunk size to 512 samples
   - Enable GPU acceleration
   - Minimize background processes

2. **For High Accuracy**:
   - Use `pathumma-large` model
   - Increase chunk overlap
   - Enable noise reduction
   - Use stable network connection

3. **For Multiple Clients**:
   - Monitor GPU memory usage
   - Adjust batch size based on clients
   - Use load balancing if needed
   - Monitor WebSocket connection limits

### Monitoring

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor server logs
tail -f server.log

# Test WebSocket connection
wscat -c ws://localhost:8765
```

## Hardware Requirements

### Minimum Requirements
- **CPU**: Intel i5 or AMD Ryzen 5
- **RAM**: 8GB DDR4
- **GPU**: NVIDIA GTX 1060 (6GB VRAM) or better
- **Storage**: 10GB free space
- **Network**: 100Mbps for model downloads

### Recommended Requirements
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **RAM**: 16GB+ DDR4
- **GPU**: NVIDIA RTX 3070 (8GB VRAM) or better
- **Storage**: SSD with 20GB free space
- **Network**: 1Gbps for fast model downloads

### GPU Memory Usage by Model

| Model | VRAM Usage | Recommended GPU |
|-------|------------|-----------------|
| biodatlab-faster | ~3GB | GTX 1060 6GB+ |
| biodatlab-medium-faster | ~5GB | RTX 3060 8GB+ |
| pathumma-large | ~7GB | RTX 3070 8GB+ |

## Security Considerations

### Production Deployment

1. **WebSocket Security**:
   - Use WSS (WebSocket Secure) in production
   - Implement authentication and authorization
   - Rate limiting for connections and messages
   - Input validation for all messages

2. **Network Security**:
   - Restrict access to specific IP ranges
   - Use reverse proxy (nginx/Apache)
   - Implement CORS policies
   - Monitor connection attempts

3. **Data Privacy**:
   - Audio data is processed in memory only
   - No persistent storage of audio/transcriptions
   - Optional audio logging for debugging
   - Comply with data protection regulations

### Example Nginx Configuration

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location /ws/control {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
    
    location /ws/audio {
        proxy_pass http://127.0.0.1:8766;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Contributing

We welcome contributions to improve the Thai ASR WebSocket server:

1. **Bug Reports**: Create issues with detailed reproduction steps
2. **Feature Requests**: Propose new features with use cases
3. **Code Contributions**: Submit pull requests with tests
4. **Documentation**: Improve docs and examples

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd MANJU/backend

# Install development dependencies
pip install -r requirements_realtime.txt
pip install -r requirements_dev.txt

# Run tests
python -m pytest tests/

# Start development server
python realtime_thai_asr_server.py
```

## License

This Thai ASR WebSocket server is part of the MANJU project and follows the same licensing terms.

## Support

For support and questions:
- **GitHub Issues**: Technical problems and bugs
- **Discussions**: Feature requests and general questions
- **Documentation**: Check this README and inline code comments

---

**Made with ❤️ for Thai language processing**
