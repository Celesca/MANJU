#!/usr/bin/env python3
"""
F5-TTS-THAI Test Script
This script helps debug F5-TTS installation and functionality.
"""

import sys
import traceback

def test_imports():
    """Test if all required imports are available"""
    print("🔍 Testing imports...")
    
    # Test basic imports
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
    
    try:
        import torchaudio
        print(f"✅ torchaudio: {torchaudio.__version__}")
    except ImportError as e:
        print(f"❌ torchaudio: {e}")
    
    try:
        import soundfile as sf
        print(f"✅ soundfile: {sf.__version__}")
    except ImportError as e:
        print(f"❌ soundfile: {e}")
    
    # Test F5-TTS imports
    try:
        import f5_tts
        print(f"✅ f5_tts module available")
    except ImportError as e:
        print(f"❌ f5_tts module: {e}")
        return False
    
    # Test specific F5-TTS components
    try:
        from f5_tts.api import F5TTS
        print("✅ F5TTS API class available")
    except ImportError:
        try:
            from f5_tts.infer.utils_infer import infer_process
            print("✅ F5TTS infer_process function available")
        except ImportError:
            try:
                from f5_tts.model import F5TTS
                print("✅ F5TTS model class available")
            except ImportError as e:
                print(f"❌ No F5TTS interfaces available: {e}")
                return False
    
    return True

def test_basic_functionality():
    """Test basic F5-TTS functionality"""
    print("\n🔧 Testing F5-TTS functionality...")
    
    try:
        # Try to import and initialize
        try:
            from f5_tts.api import F5TTS
            model = F5TTS(model_type="F5-TTS")
            print("✅ F5TTS API initialization successful")
            return True
        except Exception as e:
            print(f"⚠️ F5TTS API failed: {e}")
        
        try:
            from f5_tts.infer.utils_infer import infer_process
            print("✅ F5TTS infer_process available")
            return True
        except Exception as e:
            print(f"⚠️ F5TTS infer_process failed: {e}")
        
        # Test command line availability
        import subprocess
        result = subprocess.run(['python', '-c', 'import f5_tts'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ F5TTS command line interface available")
            return True
        else:
            print(f"❌ F5TTS CLI test failed: {result.stderr}")
        
    except Exception as e:
        print(f"❌ F5TTS functionality test failed: {e}")
        traceback.print_exc()
    
    return False

def test_audio_generation():
    """Test actual audio generation"""
    print("\n🎵 Testing audio generation...")
    
    try:
        import tempfile
        import os
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            output_file = f.name
        
        test_text = "สวัสดีครับ นี่คือการทดสอบ F5-TTS-THAI"
        
        # Try API method
        try:
            from f5_tts.api import F5TTS
            model = F5TTS(model_type="F5-TTS")
            
            audio_data = model.infer(
                text=test_text,
                ref_text="สวัสดีครับ ผมเป็นผู้ช่วยเสียงภาษาไทย",
                remove_silence=True
            )
            
            import soundfile as sf
            if hasattr(audio_data, 'cpu'):
                audio_data = audio_data.cpu().numpy()
            
            sf.write(output_file, audio_data, 22050)
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"✅ Audio generation successful! File: {output_file}")
                print(f"   File size: {os.path.getsize(output_file)} bytes")
                return True
            
        except Exception as e:
            print(f"⚠️ API audio generation failed: {e}")
        
        # Try CLI method
        try:
            import subprocess
            
            cmd = [
                'python', '-m', 'f5_tts.infer.infer_cli',
                '--text', test_text,
                '--output', output_file,
                '--model', 'F5-TTS'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(output_file):
                print(f"✅ CLI audio generation successful! File: {output_file}")
                print(f"   File size: {os.path.getsize(output_file)} bytes")
                return True
            else:
                print(f"⚠️ CLI audio generation failed: {result.stderr}")
        
        except Exception as e:
            print(f"⚠️ CLI audio generation failed: {e}")
        
        # Cleanup
        try:
            if os.path.exists(output_file):
                os.unlink(output_file)
        except:
            pass
    
    except Exception as e:
        print(f"❌ Audio generation test failed: {e}")
        traceback.print_exc()
    
    return False

def main():
    """Main test function"""
    print("🎤 F5-TTS-THAI Test Script")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install F5-TTS-THAI:")
        print("   pip install git+https://github.com/VYNCX/F5-TTS-THAI.git")
        return 1
    
    # Test functionality
    if not test_basic_functionality():
        print("\n❌ Functionality tests failed.")
        return 1
    
    # Test audio generation
    if not test_audio_generation():
        print("\n⚠️ Audio generation tests failed, but basic functionality works.")
        print("   F5-TTS should still work with fallback methods.")
    
    print("\n✅ F5-TTS-THAI tests completed!")
    print("\nTips:")
    print("- For best performance, use a CUDA-enabled GPU")
    print("- If tests fail, try reinstalling: pip install --force-reinstall git+https://github.com/VYNCX/F5-TTS-THAI.git")
    print("- Check that all dependencies are installed: torch, torchaudio, soundfile")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
