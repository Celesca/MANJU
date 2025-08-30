# 🎯 Model Selection Guide

## Multi-agent Call Center Backend - Model Selection

The backend now supports multiple Thai ASR models that users can choose from via the API. This provides flexibility to balance between speed and accuracy based on your needs.

## 🗂️ Available Models

### 🚀 Faster-Whisper Models (Optimized for Speed)

| Model ID | Name | Performance | Description |
|----------|------|-------------|-------------|
| `biodatlab-faster` ⭐ | Biodatlab Whisper Thai (Faster) | **Fast** | Thai-optimized model with 2-4x speed improvement |
| `large-v3-faster` | Whisper Large-v3 (Faster) | **Balanced** | Standard large-v3 with faster-whisper optimization |
| `medium-faster` | Whisper Medium (Faster) | **Fast** | Medium model optimized for speed |

### 🎯 Standard Whisper Models (Optimized for Accuracy)

| Model ID | Name | Performance | Description |
|----------|------|-------------|-------------|
| `pathumma-large` ⭐ | Pathumma Whisper Thai Large-v3 | **Accurate** | NECTEC's Thai-specific model |
| `large-v3-standard` | Whisper Large-v3 (Standard) | **Accurate** | OpenAI's standard large-v3 model |
| `medium-standard` | Whisper Medium (Standard) | **Balanced** | Standard medium model |

⭐ = Recommended models

## 📡 New API Endpoints

### List Available Models
```http
GET /api/models
```

**Response:**
```json
{
  "models": [
    {
      "id": "biodatlab-faster",
      "name": "Biodatlab Whisper Thai (Faster)",
      "type": "faster_whisper",
      "language": "th",
      "description": "Optimized Thai model based on large-v3, 2-4x faster performance",
      "performance_tier": "fast",
      "recommended": true
    }
  ],
  "current_model": {
    "id": "biodatlab-faster",
    "name": "Biodatlab Whisper Thai (Faster)",
    "type": "faster_whisper"
  }
}
```

### Load Specific Model
```http
POST /api/models/{model_id}/load
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/models/pathumma-large/load"
```

### Transcribe with Model Selection
```http
POST /api/asr
Content-Type: multipart/form-data

Parameters:
- file: Audio file (required)
- model_id: Model to use (default: "biodatlab-faster")
- language: Language code (default: "th")
- use_vad: Voice Activity Detection (default: true)
- beam_size: Beam size (default: 1)
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/asr" \
     -F "file=@audio.wav" \
     -F "model_id=pathumma-large" \
     -F "language=th"
```

## 💻 Usage Examples

### Using Python API Client

```python
from api_client import CallCenterAPIClient

client = CallCenterAPIClient("http://localhost:8000")

# List available models
models = client.get_available_models()
print("Available models:", [m['id'] for m in models['models']])

# Load specific model
client.load_model("pathumma-large")

# Transcribe with specific model
result = client.transcribe_audio(
    "audio.wav", 
    model_id="biodatlab-faster"
)
print(f"Transcription: {result['text']}")

# Batch transcription with model selection
results = client.transcribe_batch(
    ["audio1.wav", "audio2.wav"],
    model_id="pathumma-large"
)
```

### Using Command Line

```bash
# List available models
python api_client.py --models

# Load specific model
python api_client.py --load-model pathumma-large

# Transcribe with model selection
python api_client.py --transcribe audio.wav --model-id biodatlab-faster

# Batch transcription
python api_client.py --batch audio1.wav audio2.wav --model-id pathumma-large

# Get current model info
python api_client.py --info
```

### Using curl

```bash
# List models
curl http://localhost:8000/api/models

# Load model
curl -X POST http://localhost:8000/api/models/biodatlab-faster/load

# Transcribe with model
curl -X POST "http://localhost:8000/api/asr" \
     -F "file=@audio.wav" \
     -F "model_id=pathumma-large"
```

## 🔧 Model Selection Strategy

### For Speed (Real-time Applications)
- **Primary**: `biodatlab-faster` - Best balance of speed and accuracy for Thai
- **Alternative**: `medium-faster` - Fastest option with acceptable accuracy

### For Accuracy (High-quality Transcription)
- **Primary**: `pathumma-large` - Best accuracy for Thai language
- **Alternative**: `large-v3-standard` - High accuracy with broader language support

### For Balanced Performance
- **Primary**: `large-v3-faster` - Good balance of speed and accuracy
- **Alternative**: `medium-standard` - Reliable performance

## 🚀 Performance Comparison

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|---------|----------|
| biodatlab-faster | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Production Thai ASR |
| pathumma-large | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | High-quality Thai transcription |
| large-v3-faster | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | General purpose |
| medium-faster | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Resource-constrained environments |

## 🔄 Dynamic Model Switching

The server supports dynamic model switching without restart:

1. **Automatic Loading**: If no model is loaded, the server will automatically load the requested model
2. **On-demand Switching**: Models are switched automatically when a different model is requested
3. **Session Persistence**: The loaded model stays active for subsequent requests
4. **Memory Management**: Only one model is kept in memory at a time

## ⚙️ Configuration

### Default Model Priority
1. `biodatlab-faster` (Thai-optimized, fast)
2. `pathumma-large` (Thai-optimized, accurate)
3. `large-v3-faster` (General, fast)
4. `large-v3-standard` (General, accurate)

### Environment Variables
```bash
export DEFAULT_MODEL_ID=biodatlab-faster
export FALLBACK_MODEL_ID=large-v3-faster
```

## 🧪 Testing Model Selection

```bash
# Test all model-related functionality
python test_model_selection.py

# Test specific model loading
python -c "
from model_manager import get_model_manager
manager = get_model_manager()
models = manager.get_available_models()
print('Available models:', [m['id'] for m in models])
"
```

## 🔮 Advanced Usage

### Custom Model Configuration

```python
# Load model with custom configuration
config_overrides = {
    'beam_size': 2,
    'use_vad': False,
    'chunk_length_ms': 15000
}

manager = get_model_manager()
manager.load_model('pathumma-large', config_overrides)
```

### Model Performance Monitoring

```python
# Get detailed model information
client = CallCenterAPIClient()
model_info = client.get_asr_info()

print(f"Current model: {model_info['name']}")
print(f"Performance tier: {model_info['performance_tier']}")
print(f"Model type: {model_info['type']}")
```

This model selection system provides the flexibility to choose the best ASR model for your specific use case while maintaining a simple and consistent API interface.
