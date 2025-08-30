#!/usr/bin/env python3
"""
Setup script for faster-whisper optimization
This will install faster-whisper and test the speed improvement
"""

import subprocess
import sys
import time
import os

def install_faster_whisper():
    """Install faster-whisper and dependencies"""
    print("🚀 Installing faster-whisper for 2-4x speed improvement...")
    
    packages = [
        "faster-whisper",
        "torch",
        "torchaudio"
    ]
    
    for package in packages:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def test_faster_whisper():
    """Test faster-whisper installation"""
    print("\n🧪 Testing faster-whisper installation...")
    
    try:
        from backend.whisper.faster_whisper import WhisperModel
        print("✅ faster-whisper imported successfully")
        
        # Test model loading
        print("🔄 Testing model loading (this may take a moment)...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ Model loading test passed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def create_speed_test_script():
    """Create a script to test speed differences"""
    script_content = '''#!/usr/bin/env python3
"""
Speed comparison test between standard Whisper and faster-whisper
"""

import time
import tempfile
import os
from pydub import AudioSegment
from pydub.generators import Sine

def create_test_audio(duration_seconds=10):
    """Create a test audio file"""
    print(f"🎵 Creating {duration_seconds}s test audio...")
    
    # Generate a sine wave (simple test audio)
    sine_wave = Sine(440).to_audio_segment(duration=duration_seconds * 1000)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sine_wave.export(temp_file.name, format="wav")
    
    return temp_file.name

def test_standard_whisper(audio_file):
    """Test standard Whisper speed"""
    print("⏱️  Testing standard Whisper...")
    
    try:
        from transformers import pipeline
        import torch
        
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("automatic-speech-recognition", 
                       model="openai/whisper-tiny", 
                       device=device)
        
        start_time = time.time()
        result = pipe(audio_file)
        end_time = time.time()
        
        standard_time = end_time - start_time
        print(f"✅ Standard Whisper: {standard_time:.2f} seconds")
        return standard_time
        
    except Exception as e:
        print(f"❌ Standard Whisper failed: {e}")
        return None

def test_faster_whisper(audio_file):
    """Test faster-whisper speed"""
    print("🚀 Testing faster-whisper...")
    
    try:
        from faster_whisper import WhisperModel
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "int8_float16" if device == "cuda" else "int8"
        
        model = WhisperModel("tiny", device=device, compute_type=compute_type)
        
        start_time = time.time()
        segments, info = model.transcribe(audio_file)
        list(segments)  # Consume the generator
        end_time = time.time()
        
        faster_time = end_time - start_time
        print(f"✅ faster-whisper: {faster_time:.2f} seconds")
        return faster_time
        
    except Exception as e:
        print(f"❌ faster-whisper failed: {e}")
        return None

def main():
    """Run speed comparison"""
    print("🏁 Whisper Speed Comparison Test")
    print("=" * 40)
    
    # Create test audio
    audio_file = create_test_audio(10)
    
    try:
        # Test both versions
        standard_time = test_standard_whisper(audio_file)
        faster_time = test_faster_whisper(audio_file)
        
        # Compare results
        if standard_time and faster_time:
            speedup = standard_time / faster_time
            print(f"\\n📊 Results:")
            print(f"   Standard Whisper: {standard_time:.2f}s")
            print(f"   faster-whisper:   {faster_time:.2f}s")
            print(f"   🚀 Speedup: {speedup:.1f}x faster!")
            
            if speedup > 1.5:
                print("✅ Significant speed improvement detected!")
            else:
                print("⚠️  Limited improvement - check GPU setup")
        
    finally:
        # Cleanup
        try:
            os.unlink(audio_file)
        except:
            pass

if __name__ == "__main__":
    main()
'''
    
    with open("speed_test.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("📄 Created speed_test.py - run it to compare performance!")

def main():
    """Main setup function"""
    print("🚀 faster-whisper Setup for Thai ASR")
    print("=" * 40)
    
    # Install packages
    if not install_faster_whisper():
        print("❌ Installation failed!")
        return
    
    # Test installation
    if not test_faster_whisper():
        print("❌ Testing failed!")
        return
    
    # Create test script
    create_speed_test_script()
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: python speed_test.py (to see speed improvement)")
    print("2. In your code, set use_faster_whisper=True in ProcessingConfig")
    print("3. Enjoy 2-4x faster ASR performance! 🚀")
    
    print("\n💡 Optimization tips:")
    print("• Use compute_type='int8_float16' for GPU")
    print("• Set beam_size=1 for maximum speed")
    print("• Enable VAD to skip silent parts")
    print("• Use larger chunk sizes (30s) for better efficiency")

if __name__ == "__main__":
    main()
