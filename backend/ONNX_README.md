# ONNX Server Setup and Usage

## Overview
The ONNX server provides faster inference for Whisper models using Sherpa-ONNX optimized runtime.

## Installation

1. Install dependencies:
```bash
pip install sherpa-onnx>=1.9.0 onnxruntime>=1.16.0 soundfile scipy
```

2. Start the ONNX server:
```bash
# Windows
start_onnx_server.bat

# Manual
python onnx_server.py
```

## Available Models

### Whisper Large-v3 ONNX (Recommended)
- Model ID: `whisper-large-v3-onnx`
- Performance: ~2-5x faster than PyTorch
- Language: Thai optimized
- Files auto-downloaded from HuggingFace

### Whisper Medium ONNX
- Model ID: `whisper-medium-onnx`
- Performance: Balanced speed/accuracy
- Smaller memory footprint

## API Endpoints

### Base URL: `http://localhost:8001`

#### POST `/api/onnx/asr`
Transcribe audio using ONNX models.

**Form Data:**
- `file`: Audio file (required)
- `model_id`: ONNX model ID (default: whisper-large-v3-onnx)
- `language`: Language code (default: th)

#### GET `/api/onnx/models`
List available ONNX models.

#### POST `/api/onnx/models/{model_id}/load`
Load specific ONNX model.

#### GET `/api/onnx/info`
Get current loaded model information.

#### GET `/onnx/health`
Health check with Sherpa-ONNX status.

## Example Usage

### cURL
```bash
curl -X POST "http://localhost:8001/api/onnx/asr" \
  -F "file=@audio.wav" \
  -F "model_id=whisper-large-v3-onnx" \
  -F "language=th"
```

### Python
```python
import requests

files = {'file': open('audio.wav', 'rb')}
data = {
    'model_id': 'whisper-large-v3-onnx',
    'language': 'th'
}

response = requests.post('http://localhost:8001/api/onnx/asr', files=files, data=data)
result = response.json()
print(result['text'])
```

### Postman
1. Method: POST
2. URL: `http://localhost:8001/api/onnx/asr`
3. Body: form-data
   - file: [Upload audio file]
   - model_id: whisper-large-v3-onnx
   - language: th

## Performance Comparison

| Engine | Speed | Memory | Accuracy |
|--------|-------|--------|----------|
| PyTorch Whisper | 1x | High | High |
| Faster-Whisper | 2-4x | Medium | High |
| **Sherpa-ONNX** | **2-5x** | **Low** | **High** |

## Model Storage

Models are downloaded to:
```
models/onnx/
├── whisper-large-v3/
│   ├── encoder.onnx
│   ├── decoder.onnx
│   └── tokens.txt
└── whisper-medium/
    ├── encoder.onnx
    ├── decoder.onnx
    └── tokens.txt
```

## Troubleshooting

### Sherpa-ONNX not found
```bash
pip install sherpa-onnx
```

### CUDA Support
For GPU acceleration:
```bash
pip install onnxruntime-gpu
```

### Model Download Issues
Check internet connection and disk space. Models are ~1-3GB each.

## Configuration

Environment variables:
- `ONNX_PORT`: Server port (default: 8001)
- `HOST`: Bind address (default: 0.0.0.0)
- `DEBUG`: Enable debug mode (default: false)

## Integration

The ONNX server runs independently on port 8001, separate from the main server on port 8000. This allows:
- A/B testing between engines
- Load balancing
- Independent scaling
- Fallback scenarios
