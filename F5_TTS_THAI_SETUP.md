# F5-TTS-THAI (VIZINTZOR) Integration Guide

This guide explains how to set up and use the authentic Thai F5-TTS model with the VIZINTZOR pre-trained checkpoint for high-quality Thai text-to-speech.

## 🎯 What Changed

The chatbot now uses the **VIZINTZOR F5-TTS-THAI** pre-trained model with proper DiT model loading instead of the generic F5-TTS model. This provides:

- ✅ **Authentic Thai pronunciation** (no more Chinese accent!)
- ✅ **Natural Thai voice** trained on Thai datasets
- ✅ **Proper model loading** using DiT architecture and load_model function
- ✅ **Better Thai language understanding**
- ✅ **High-quality 24kHz audio output**

## 📋 Prerequisites

1. **Python 3.10+** with pip
2. **CUDA GPU** (recommended for best performance)
3. **~2GB free disk space** for the model
4. **Stable internet connection** for downloading

## 🚀 Quick Setup

### Step 1: Automated Installation (Recommended)

Run the complete installation script:
```bash
python install_f5_tts_thai.py
```

This will automatically:
- Install all dependencies (PyTorch, TorchAudio, etc.)
- Clone and install F5-TTS-THAI from source
- Download the VIZINTZOR Thai model checkpoint
- Verify the installation

### Step 2: Test Setup

```bash
python test_thai_model.py
```

### Step 4: Run Chatbot

```bash
streamlit run voice_chatbot.py
```

## 🔧 Configuration

The Thai model is automatically detected when these files are present:

```
project/
├── models/
│   ├── model_1000000.pt    # Main Thai model (required)
│   ├── vocab.txt           # Thai vocabulary (optional)
│   └── config.json         # Model config (optional)
├── voice_chatbot.py
├── setup_thai_model.py
└── test_thai_model.py
```

## 🎵 Usage

1. **Start the chatbot**: `streamlit run voice_chatbot.py`
2. **Open sidebar** → TTS settings
3. **Select engine**: Choose "F5-TTS-THAI" 
4. **Verify status**: Check the debug panel shows:
   - ✅ Available: True
   - ✅ Thai model path: models/model_1000000.pt
   - ✅ Model initialized: True

## 🧪 Testing

### Test the Installation
```bash
python test_thai_model.py
```

### Test in Chatbot
1. Go to sidebar → F5-TTS-THAI Debug Info
2. Click "🧪 Test F5-TTS-THAI (VIZINTZOR)"
3. Listen for authentic Thai voice

### Sample Test Text
```
สวัสดีครับ นี่คือการทดสอบเสียงไทยจากโมเดล VIZINTZOR ที่มีการออกเสียงภาษาไทยที่ถูกต้องและธรรมชาติ
```

## 🔍 Troubleshooting

### Problem: "Thai model checkpoint not found"
**Solution:**
```bash
# Download the model
python setup_thai_model.py

# Or check these locations:
ls models/model_1000000.pt
ls model_1000000.pt
ls ckpts/model_1000000.pt
```

### Problem: "Chinese accent in Thai speech"
**Cause:** Using generic F5-TTS instead of Thai model  
**Solution:** Ensure `model_1000000.pt` is downloaded and detected

### Problem: "CUDA out of memory"
**Solutions:**
1. **Reduce model usage**: Use gTTS instead for long texts
2. **Free GPU memory**: Close other GPU applications
3. **CPU fallback**: Model will automatically use CPU

### Problem: "Model loading failed"
**Solutions:**
1. **Re-download model**: `python setup_thai_model.py`
2. **Check file size**: Model should be ~1350 MB
3. **Verify integrity**: Delete and re-download if corrupted

### Problem: "Import errors"
**Solution:**
```bash
# Reinstall F5-TTS-THAI
pip uninstall f5-tts
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git

# Install missing dependencies
pip install torch torchaudio soundfile
```

## 📊 Performance

| Setting | Speed | Quality | GPU Memory |
|---------|-------|---------|------------|
| F5-TTS-THAI | Slow | Excellent | ~2GB |
| gTTS | Fast | Good | None |
| pyttsx3 | Fast | Poor (Thai) | None |

**Recommendation:** Use F5-TTS-THAI for important responses, gTTS for quick interactions.

## 🎛️ Advanced Configuration

### Model Parameters
The system uses these optimized settings for Thai:

```python
{
    "speed": 0.9,           # Slightly slower for clear Thai
    "nfe_step": 32,         # Quality vs speed balance
    "cfg_strength": 2.0,    # Classifier-free guidance
    "remove_silence": True, # Clean output
}
```

### Custom Reference Audio
To improve voice quality, place Thai reference audio:
```
temp/thai_reference.wav  # Auto-generated from gTTS
```

### Model Variants
- `model_1000000.pt` - Latest (recommended)
- `old_small_model/model_850000.pt` - Smaller, faster
- `model/model_50000.pt` - Archived version

## 🔗 Resources

- **Model Source**: [VIZINTZOR/F5-TTS-THAI](https://huggingface.co/VIZINTZOR/F5-TTS-THAI)
- **F5-TTS-THAI Repo**: [VYNCX/F5-TTS-THAI](https://github.com/VYNCX/F5-TTS-THAI)
- **Original F5-TTS**: [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
- **Colab Demo**: [F5-TTS-THAI Colab](https://colab.research.google.com/drive/10yb4-mGbSoyyfMyDX1xVF6uLqfeoCNxV)

## 🆘 Support

If you encounter issues:

1. **Check logs** in the Streamlit sidebar debug panel
2. **Run test script**: `python test_thai_model.py`
3. **Verify installation**: Ensure all files are downloaded
4. **Fallback option**: The system automatically falls back to gTTS if F5-TTS fails

The chatbot will display detailed status information and error messages to help diagnose issues.

---

**🎉 Enjoy authentic Thai text-to-speech with the VIZINTZOR model!**
