# F5-TTS-THAI Implementation Guide

## Overview

The voice chatbot now includes a fully functional F5-TTS-THAI implementation for premium Thai text-to-speech generation. This provides the highest quality Thai voice synthesis available.

## Implementation Details

### Multi-Method Approach

The F5-TTS-THAI integration uses three different methods to maximize compatibility:

1. **API-based approach**: Direct use of F5TTS class
2. **Function-based approach**: Using `infer_process` function
3. **Command-line approach**: Fallback to CLI execution

### Smart Fallback System

```
F5-TTS-THAI (Premium) → gTTS (Reliable) → pyttsx3 (Basic)
```

If F5-TTS fails for any reason, the system automatically falls back to gTTS to ensure users always get audio output.

## Installation

### Method 1: Automatic Script
```bash
install_f5_tts.bat
```

### Method 2: Manual Installation
```bash
# Install PyTorch with CUDA support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install F5-TTS-THAI
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git

# Additional dependencies
pip install soundfile librosa
```

## Features

### Advantages of F5-TTS-THAI
- **Highest Quality**: Superior audio quality compared to gTTS or pyttsx3
- **Natural Sounding**: More natural Thai pronunciation and intonation
- **Offline Capability**: Works without internet connection
- **Voice Cloning**: Can adapt to custom reference voices
- **GPU Acceleration**: Leverages CUDA for faster generation

### Smart Features
- **Reference Text**: Uses default Thai reference for consistency
- **Multiple Output Formats**: Supports WAV and other audio formats
- **Automatic Cleanup**: Manages temporary files automatically
- **Progress Indicators**: Shows generation status in the UI
- **Error Handling**: Graceful fallback on any failure

## Usage

### Automatic Selection
When "auto" engine is selected, F5-TTS-THAI is preferred for Thai text:
```python
st.session_state.tts.speak(text, engine="auto")
```

### Manual Selection
Force F5-TTS-THAI usage:
```python
st.session_state.tts.speak(text, engine="F5-TTS-THAI")
```

## Debugging

### Debug Panel
The sidebar includes a debug panel for F5-TTS-THAI that shows:
- Installation status
- Model initialization status
- Available methods
- Test functionality

### Test F5-TTS
Use the "🧪 Test F5-TTS-THAI" button to verify the installation:
- Generates a sample Thai phrase
- Shows success/failure status
- Helps diagnose issues

## Troubleshooting

### Common Issues

1. **"F5-TTS not available"**
   - Install F5-TTS-THAI: `pip install git+https://github.com/VYNCX/F5-TTS-THAI.git`
   - Check PyTorch installation: `python -c "import torch; print(torch.version)"`

2. **"All methods failed"**
   - Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
   - Verify F5-TTS import: `python -c "import f5_tts"`
   - Falls back to gTTS automatically

3. **Slow generation**
   - Enable GPU acceleration by installing CUDA
   - F5-TTS works on CPU but is slower
   - Consider using gTTS for faster results

### Performance Optimization

**For Best Performance:**
- Install CUDA-enabled PyTorch
- Use GPU with sufficient VRAM (4GB+)
- Keep reference texts short

**For Compatibility:**
- CPU-only installation works but is slower
- gTTS provides faster alternative
- pyttsx3 as last resort

## Configuration

### Default Settings
```python
# Default reference text (customizable)
ref_text = "สวัสดีครับ ผมเป็นผู้ช่วยเสียงภาษาไทย"

# Audio settings
sample_rate = 22050  # Standard for F5-TTS
format = "WAV"       # Best compatibility
```

### Customization Options
- **Reference Audio**: Provide custom voice samples
- **Reference Text**: Customize pronunciation guide
- **Output Format**: Change audio format if needed
- **Sample Rate**: Adjust quality vs. file size

## Integration with Voice Chatbot

### Seamless Operation
1. **Voice Input**: Record or upload audio
2. **ASR Processing**: Whisper transcribes Thai speech
3. **LLM Response**: OpenRouter generates Thai response
4. **TTS Output**: F5-TTS-THAI speaks the response
5. **Fallback**: gTTS if F5-TTS unavailable

### UI Integration
- **Engine Selection**: Choose F5-TTS-THAI from dropdown
- **Status Display**: Shows F5-TTS availability
- **Progress Indicators**: Real-time generation status
- **Audio Player**: Streamlit audio component
- **Debug Panel**: Troubleshooting information

## Technical Details

### File Management
- **Temporary Files**: Auto-generated and cleaned up
- **Audio Format**: WAV for best compatibility
- **Cleanup Timer**: 10-second delay for proper playback
- **Thread Safety**: Background cleanup threads

### Error Handling
- **Graceful Degradation**: Falls back to gTTS on failure
- **User Feedback**: Clear status messages
- **Silent Failures**: No disruption to chat flow
- **Detailed Logging**: Debug information available

## Performance Comparison

| Engine | Quality | Speed | Thai Support | Offline | GPU |
|--------|---------|-------|--------------|---------|-----|
| F5-TTS-THAI | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ✅ |
| gTTS | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | ❌ |
| pyttsx3 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ | ❌ |

## Conclusion

The F5-TTS-THAI implementation provides premium Thai text-to-speech capabilities while maintaining compatibility and reliability through intelligent fallback mechanisms. Users get the best possible audio quality when F5-TTS is available, with seamless degradation to other engines when needed.
