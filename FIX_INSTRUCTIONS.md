# 🔧 Fix Instructions for Voice Chatbot Issues

## Problem 1: FFmpeg Missing Error

You're seeing this error:
```
Error in batch processing: ffmpeg was not found but is required to load audio files from filename
Transcription failed or empty
```

### 🚀 Quick Fix for FFmpeg

**Option 1: Run the FFmpeg installer**
```cmd
install_ffmpeg.bat
```

**Option 2: Manual Installation**
```cmd
# Using winget (Windows 10/11)
winget install "FFmpeg (Essentials Build)"

# Using chocolatey
choco install ffmpeg

# Manual download
# 1. Download from: https://github.com/BtbN/FFmpeg-Builds/releases
# 2. Extract to C:\ffmpeg
# 3. Add C:\ffmpeg\bin to PATH
```

**Option 3: Portable FFmpeg**
```cmd
# Download from: https://www.gyan.dev/ffmpeg/builds/
# Set FFMPEG_BINARY environment variable to ffmpeg.exe path
```

## Problem 2: Keras/TensorFlow Compatibility Error

You're seeing this error:
```
RuntimeError: Failed to import transformers.models.whisper.modeling_tf_whisper because of the following error: Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers. Please install the backwards-compatible tf-keras package with `pip install tf-keras`.
```

## 🚀 Quick Fix (Recommended)

**Option 1: Run the comprehensive fix script**
```cmd
python fix_all_issues.py
```

**Option 2: Manual fix commands**
```cmd
# Clean installations
pip uninstall torch torchvision torchaudio tensorflow keras tf-keras -y

# Install compatibility layer
pip install tf-keras

# Install PyTorch CPU-only (stable)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
```

## 📋 Step-by-Step Solution

### Step 1: Clean Environment
```cmd
pip uninstall torch torchvision torchaudio -y
pip uninstall tensorflow keras tf-keras -y
pip cache purge
```

### Step 2: Install Compatibility Packages
```cmd
pip install tf-keras
```

### Step 3: Install Stable PyTorch
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install Requirements
```cmd
pip install -r requirements.txt
```

### Step 5: Test Installation
```cmd
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers: OK')"
```

## 🛡️ Alternative: Use Simple Chatbot

If you want to skip the ASR issues for now:

```cmd
streamlit run simple_chatbot.py
```

This version works without PyTorch and provides:
- ✅ Text chat with Ollama
- ✅ Text-to-speech responses
- ✅ Conversation history
- ❌ No voice recognition (text input only)

## 🔍 What Causes This Issue?

1. **Keras 3 Compatibility**: Transformers library expects older Keras
2. **TensorFlow/PyTorch Conflicts**: Mixed framework installations
3. **Windows DLL Issues**: Missing Visual C++ components

## 🎯 What the Fix Does

1. **Removes Conflicts**: Cleans all TensorFlow/PyTorch installations
2. **Installs tf-keras**: Provides backward compatibility for Transformers
3. **Stable PyTorch**: CPU-only version (more stable on Windows)
4. **Forces PyTorch Backend**: Environment variables prevent TensorFlow use

## 📊 Verification Commands

After running the fix, test with:

```cmd
python -c "
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import torch
from transformers import pipeline
print('✅ All imports successful!')
print(f'PyTorch: {torch.__version__}')
"
```

## 🚨 If Fix Doesn't Work

1. **Install Visual C++ Redistributable**
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install and restart computer

2. **Use Simple Chatbot**
   ```cmd
   streamlit run simple_chatbot.py
   ```

3. **Check Ollama**
   ```cmd
   ollama serve
   ollama pull phi3
   ```

## 🎉 Success Indicators

When everything works, you should see:
- ✅ PyTorch imports without DLL errors
- ✅ Transformers loads with PyTorch backend
- ✅ Streamlit starts without crashes
- ✅ Voice recording works in browser
- ✅ ASR transcription works
- ✅ Ollama responds to queries
- ✅ TTS speaks responses

## 📞 Final Test

```cmd
streamlit run voice_chatbot.py
```

Open browser → Record voice → See transcription → Get AI response → Hear TTS

## 💡 Pro Tips

1. **Always use CPU PyTorch** for Windows stability
2. **Set environment variables** to force PyTorch backend
3. **Keep tf-keras installed** for Transformers compatibility
4. **Use simple_chatbot.py** as fallback during development
5. **Restart terminal** after running fixes

---

## 🔧 Files for Different Scenarios

- `fix_all_issues.py` - Comprehensive fix (recommended)
- `simple_chatbot.py` - Text-only version (no ASR)
- `voice_chatbot.py` - Full version with voice features
- `setup_advanced.bat` - Windows automated setup
