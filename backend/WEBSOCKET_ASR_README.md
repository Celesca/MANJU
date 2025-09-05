# WebSocket Thai ASR Server

A high-performance WebSocket-based Thai speech recognition server optimized with RealtimeSTT techniques for low-latency, real-time transcription.

## 🚀 Features

### Core Features
- **Real-time streaming transcription** via WebSocket
- **RealtimeSTT optimizations** for maximum performance
- **Multi-model support** with hot model switching
- **Enhanced VAD (Voice Activity Detection)** with multi-method approach
- **Concurrent transcription** support (up to 6 parallel transcriptions)
- **Performance monitoring** and health checks
- **Smart audio buffering** with backpressure control

### Performance Optimizations
- **Reduced latency**: 512 sample chunks (vs 1024)
- **Enhanced VAD**: RMS + Zero Crossing Rate + Peak detection
- **Audio normalization**: Real-time preprocessing
- **Thread pool optimization**: Dedicated worker threads
- **Smart buffering**: Prevents audio dropouts
- **Confidence scoring**: Quality-based filtering

## 📋 Requirements

- Python 3.8+
- CUDA GPU (recommended for best performance)
- FFmpeg (for audio processing)
- Required Python packages (see `requirements.txt`)

## 🏃‍♂️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python server_websocket.py
```

### 3. Connect via WebSocket
- **Control Server**: `ws://localhost:8765` (model management, stats)
- **Audio Server**: `ws://localhost:8766` (transcription, streaming)

## 🔌 WebSocket API

### Control Server (Port 8765)

#### Load Model
```javascript
{
  "type": "load_model",
  "model_id": "biodatlab-medium-faster"
}
```

#### Get Available Models
```javascript
{
  "type": "get_models"
}
```

#### Get Server Statistics
```javascript
{
  "type": "get_stats"
}
```

#### Health Check
```javascript
{
  "type": "health_check"
}
```

### Audio Server (Port 8766)

#### Single Transcription
```javascript
{
  "type": "transcribe",
  "audio_data": "base64_encoded_audio",
  "config": {
    "language": "th",
    "beam_size": 1
  }
}
```

#### Start Streaming Session
```javascript
{
  "type": "start_stream",
  "session_id": "session_123",
  "config": {
    "language": "th",
    "beam_size": 1
  }
}
```

#### Send Audio Chunk (Streaming)
```javascript
{
  "type": "audio_chunk",
  "session_id": "session_123",
  "audio_data": "base64_encoded_chunk"
}
```

#### End Streaming Session
```javascript
{
  "type": "end_stream",
  "session_id": "session_123"
}
```

## 📊 Response Formats

### Transcription Result
```javascript
{
  "type": "transcription_result",
  "result": {
    "text": "สวัสดีครับ",
    "language": "th",
    "duration": 2.5,
    "processing_time": 0.3,
    "speed_ratio": 8.3,
    "real_time_factor": 0.12,
    "confidence": 0.85,
    "model": "biodatlab-medium-faster",
    "device": "cuda",
    "status": "success",
    "timestamp": "2025-09-06T10:30:00"
  }
}
```

### Real-time Transcription (Streaming)
```javascript
{
  "type": "realtime_transcription",
  "session_id": "session_123",
  "result": {
    "text": "สวัสดี",
    "confidence": 0.78,
    "timestamp": "2025-09-06T10:30:01"
  }
}
```

### Server Statistics
```javascript
{
  "type": "server_stats",
  "stats": {
    "server": {
      "uptime_seconds": 3600.5,
      "uptime_formatted": "1.0h",
      "connected_clients": {
        "control": 2,
        "audio": 1,
        "total": 3
      }
    },
    "performance": {
      "total_requests": 150,
      "real_time_factor": 0.15,
      "successful_transcriptions": 148,
      "avg_processing_time": 0.25
    }
  }
}
```

## 🎯 Supported Models

| Model ID | Description | Performance | Recommended |
|----------|-------------|-------------|-------------|
| `biodatlab-faster` | Fast Thai model | High | ✅ |
| `biodatlab-medium-faster` | Balanced performance | High | ✅ |
| `pathumma-large` | High accuracy | Medium | ✅ |
| `large-v3-faster` | Large model optimized | Medium | |
| `medium-faster` | Medium model optimized | High | |

## ⚙️ Configuration

### Audio Settings
```python
SAMPLE_RATE = 16000      # 16kHz optimal for ASR
CHUNK_SIZE = 512         # Reduced for lower latency
CHANNELS = 1             # Mono audio
BUFFER_SIZE = 4096       # Optimized buffer size
```

### Performance Settings
```python
MAX_CONCURRENT_TRANSCRIPTIONS = 6  # Parallel processing
INT16_MAX_ABS_VALUE = 32768.0      # Audio normalization
```

### VAD Settings
```python
vad_threshold = 0.015    # More sensitive detection
silence_threshold = 12   # Faster auto-stop
min_speech_chunks = 6    # Faster speech detection
```

## 📈 Performance Metrics

### Real-time Factor (RTF)
- **< 0.5**: Excellent performance
- **0.5 - 1.0**: Good performance
- **1.0 - 1.5**: Acceptable performance
- **> 1.5**: Performance warning
- **> 2.0**: Critical performance issue

### Key Metrics Tracked
- **Total Requests**: Number of transcription requests
- **Real-time Factor**: Processing time vs audio duration
- **Success Rate**: Successful vs failed transcriptions
- **Concurrent Requests**: Current parallel transcriptions
- **Buffer Health**: Audio buffer utilization
- **VAD Activations**: Voice activity detection events

## 🔧 Advanced Features

### Dynamic Model Switching
Switch models on-the-fly without restarting the server:
```javascript
{
  "type": "load_model",
  "model_id": "pathumma-large"
}
```

### Performance Monitoring
Real-time performance monitoring and health checks:
```javascript
{
  "type": "get_stats"
}
```

### Streaming Transcription
Real-time streaming with continuous transcription:
1. Start stream with session ID
2. Send audio chunks as they arrive
3. Receive real-time and final transcriptions
4. End stream when finished

### Audio Quality Enhancement
- **Anti-aliasing filtering**: Prevents audio artifacts
- **Normalization**: Consistent audio levels
- **Resampling**: High-quality sample rate conversion
- **Noise reduction**: Improved speech clarity

## 🌐 Production Deployment

### Using ngrok for External Access
```bash
# Install ngrok
npm install -g ngrok

# Expose control server
ngrok http 8765

# Expose audio server
ngrok http 8766
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8765 8766

CMD ["python", "server_websocket.py"]
```

### Systemd Service
```ini
[Unit]
Description=WebSocket Thai ASR Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/app
ExecStart=/path/to/python server_websocket.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## 🐛 Troubleshooting

### Common Issues

#### High Latency
```
Solution:
- Reduce CHUNK_SIZE in configuration
- Use faster model (biodatlab-faster)
- Enable GPU acceleration
- Check network connection
```

#### Memory Issues
```
Solution:
- Reduce MAX_CONCURRENT_TRANSCRIPTIONS
- Monitor buffer health via stats
- Use smaller models
- Increase system memory
```

#### Connection Issues
```
Solution:
- Check firewall settings
- Verify WebSocket ports are open
- Use ngrok for external access
- Check client connection code
```

### Performance Tuning

#### For Low Latency
- Use `biodatlab-faster` model
- Reduce chunk size to 256
- Enable GPU acceleration
- Minimize background processes

#### For High Accuracy
- Use `pathumma-large` model
- Increase beam size to 3
- Enable VAD filtering
- Use stable network connection

#### For Multiple Clients
- Monitor GPU memory usage
- Adjust concurrent transcription limit
- Use load balancing if needed
- Monitor WebSocket connection limits

## 📊 Monitoring

### Real-time Monitoring
```bash
# Monitor server logs
tail -f server.log

# Check GPU usage
nvidia-smi -l 1

# Monitor WebSocket connections
netstat -tlnp | grep :8765
netstat -tlnp | grep :8766
```

### Health Endpoints
- **Health Check**: Send `{"type": "health_check"}` to control server
- **Statistics**: Send `{"type": "get_stats"}` to control server
- **Model Info**: Send `{"type": "get_models"}` to control server

## 🔄 API Evolution

### Version 1.0.0 Features
- ✅ WebSocket-based transcription
- ✅ RealtimeSTT optimizations
- ✅ Multi-model support
- ✅ Streaming transcription
- ✅ Performance monitoring
- ✅ Enhanced VAD
- ✅ Concurrent processing

### Planned Features
- 🔄 Batch transcription support
- 🔄 Audio file upload via WebSocket
- 🔄 Custom model support
- 🔄 Advanced audio preprocessing
- 🔄 Multi-language support
- 🔄 Voice activity detection tuning

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the performance metrics
- Monitor server logs for errors

---

*Built with RealtimeSTT optimizations for maximum performance and reliability.*
