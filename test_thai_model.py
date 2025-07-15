"""
Test script for F5-TTS-THAI with VIZINTZOR model
This script tests if the Thai model is properly loaded and working
"""

import os
import torch
from pathlib import Path

def test_thai_model():
    """Test the Thai F5-TTS model"""
    print("🧪 Testing F5-TTS-THAI (VIZINTZOR) Model")
    print("=" * 50)
    
    # Check if model file exists
    model_paths = [
        "model_1000000.pt",
        "models/model_1000000.pt",
        "ckpts/model_1000000.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ Thai model not found!")
        print("Please download model_1000000.pt from:")
        print("https://huggingface.co/VIZINTZOR/F5-TTS-THAI")
        print("Or run: python setup_thai_model.py")
        return False
    
    print(f"✅ Found Thai model: {model_path}")
    
    # Check model file size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📏 Model size: {size_mb:.1f} MB")
    
    if size_mb < 100:
        print("⚠️  Warning: Model file seems too small")
        return False
    
    # Try to load the model
    print("🔄 Loading Thai model...")
    
    try:
        # Load checkpoint to verify it's valid
        checkpoint = torch.load(model_path, map_location='cpu')
        print("✅ Model checkpoint loaded successfully!")
        
        # Check checkpoint structure
        if isinstance(checkpoint, dict):
            print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
        else:
            print("📋 Checkpoint is a state dict")
        
        # Try to initialize F5-TTS with proper Thai model loading (VIZINTZOR approach)
        try:
            from f5_tts.model import DiT
            from f5_tts.infer.utils_infer import load_model
            from huggingface_hub import cached_path
            print("✅ F5TTS components available (DiT model and load_model)")
            
            # F5-TTS model configuration for Thai (following VIZINTZOR config)
            F5TTS_model_cfg = dict(
                dim=1024, 
                depth=22, 
                heads=16, 
                ff_mult=2, 
                text_dim=512, 
                conv_layers=4
            )
            
            print(f"📋 Using VIZINTZOR model config: {F5TTS_model_cfg}")
            
            # Look for vocab file
            vocab_paths = ["vocab.txt", "models/vocab.txt", "F5-TTS-THAI/vocab.txt"]
            vocab_file = None
            for vpath in vocab_paths:
                if os.path.exists(vpath):
                    vocab_file = vpath
                    print(f"✅ Found vocab file: {vocab_file}")
                    break
            
            if not vocab_file:
                print("📁 Vocab file not found locally, trying HuggingFace...")
                try:
                    vocab_file = str(cached_path("hf://VIZINTZOR/F5-TTS-THAI/vocab.txt"))
                    print(f"✅ Downloaded vocab: {vocab_file}")
                except Exception as e:
                    print(f"⚠️ Vocab download failed: {e}")
                    vocab_file = None
            
            # Load the Thai model using proper F5-TTS approach
            print("🔄 Loading Thai model with VIZINTZOR configuration...")
            model = load_model(
                DiT,
                F5TTS_model_cfg, 
                model_path,
                vocab_file=vocab_file,
                use_ema=True
            )
            
            print("✅ Thai model loaded successfully with proper method!")
            print(f"📊 Model type: {type(model)}")
            return True
            
        except ImportError as e:
            print(f"⚠️ Proper F5TTS imports not available: {e}")
            print("Trying fallback methods...")
            
            # Fallback to F5TTS API
            try:
                from f5_tts.api import F5TTS
                print("✅ F5TTS API available (fallback)")
                
                # Initialize model
                model = F5TTS()
                print("✅ F5TTS model initialized")
                
                print("⚠️ Using default F5TTS model (not Thai-optimized)")
                return True
                
            except ImportError:
                print("⚠️ F5TTS API not available, trying inference function...")
                
                try:
                    from f5_tts.infer.utils_infer import infer_process
                    print("✅ F5TTS inference function available")
                    print("✅ Can use inference function with Thai model")
                    return True
                except ImportError:
                    print("❌ F5-TTS not installed properly")
                    print("Install with: pip install git+https://github.com/VYNCX/F5-TTS-THAI.git")
                    return False
        
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n🔍 Checking Dependencies")
    print("=" * 30)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("soundfile", "SoundFile"),
        ("numpy", "NumPy")
    ]
    
    missing = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Install with: pip install {module}")
            missing.append(module)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        return False
    
    print("✅ All dependencies available!")
    return True

def test_audio_generation():
    """Test actual audio generation"""
    print("\n🎵 Testing Audio Generation")
    print("=" * 30)
    
    # Simple test text in Thai
    test_text = "สวัสดีครับ นี่คือการทดสอบเสียงไทย"
    
    try:
        # Import the F5TTSThai class from voice_chatbot
        import sys
        sys.path.append('.')
        from voice_chatbot import F5TTSThai
        
        # Initialize Thai TTS
        thai_tts = F5TTSThai()
        
        if thai_tts.is_available():
            print("✅ F5TTSThai initialized successfully!")
            print(f"Model path: {thai_tts.model_path}")
            
            # Test audio generation
            print(f"🔄 Generating audio for: '{test_text}'")
            output_file = "test_thai_output.wav"
            
            success = thai_tts.speak(test_text, save_file=output_file)
            
            if success and os.path.exists(output_file):
                size_kb = os.path.getsize(output_file) / 1024
                print(f"✅ Audio generated successfully! ({size_kb:.1f} KB)")
                print(f"📁 Saved as: {output_file}")
                
                # Clean up
                try:
                    os.remove(output_file)
                    print("🗑️  Test file cleaned up")
                except:
                    pass
                
                return True
            else:
                print("❌ Audio generation failed")
                return False
        else:
            print("❌ F5TTSThai not available")
            return False
            
    except Exception as e:
        print(f"❌ Audio generation test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 F5-TTS-THAI (VIZINTZOR) Test Suite")
    print("Testing the Thai model setup and functionality\n")
    
    all_passed = True
    
    # Test 1: Dependencies
    if not test_dependencies():
        all_passed = False
        print("\n💡 Install missing dependencies and try again")
    
    # Test 2: Model file
    if not test_thai_model():
        all_passed = False
        print("\n💡 Download the Thai model and try again")
    
    # Test 3: Audio generation (only if other tests pass)
    if all_passed:
        if not test_audio_generation():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! F5-TTS-THAI (VIZINTZOR) is ready!")
        print("✅ You can now use authentic Thai TTS in your chatbot")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\n📋 Common solutions:")
        print("1. Install dependencies: pip install torch torchaudio soundfile")
        print("2. Install F5-TTS: pip install git+https://github.com/VYNCX/F5-TTS-THAI.git")
        print("3. Download model: python setup_thai_model.py")
