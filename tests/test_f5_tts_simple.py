#!/usr/bin/env python3
"""
Simple F5-TTS-THAI Test Script
Tests the actual F5-TTS-THAI installation and identifies correct import paths
"""

import sys
import os
import tempfile

def test_basic_import():
    """Test basic F5-TTS import"""
    print("=== Testing Basic F5-TTS Import ===")
    try:
        import f5_tts
        print("✅ f5_tts base module imported successfully")
        print(f"   Module path: {f5_tts.__file__}")
        print(f"   Available attributes: {[attr for attr in dir(f5_tts) if not attr.startswith('_')]}")
        return True
    except ImportError as e:
        print(f"❌ f5_tts import failed: {e}")
        return False

def test_infer_utils():
    """Test inference utilities import"""
    print("\n=== Testing Inference Utils ===")
    try:
        from f5_tts.infer.utils_infer import infer_process
        print("✅ infer_process function imported successfully")
        print(f"   Function: {infer_process}")
        return infer_process
    except ImportError as e:
        print(f"❌ infer_process import failed: {e}")
        return None

def test_api_import():
    """Test API imports"""
    print("\n=== Testing API Import ===")
    try:
        from f5_tts.api import F5TTS
        print("✅ F5TTS API class imported successfully")
        print(f"   Class: {F5TTS}")
        return F5TTS
    except ImportError as e:
        print(f"❌ F5TTS API import failed: {e}")
        return None

def test_cli_availability():
    """Test CLI availability"""
    print("\n=== Testing CLI Availability ===")
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-c', 'import f5_tts.infer.infer_cli'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CLI module is available")
            return True
        else:
            print(f"❌ CLI module failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n=== Testing Dependencies ===")
    dependencies = ['torch', 'torchaudio', 'soundfile']
    all_available = True
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} is available")
        except ImportError:
            print(f"❌ {dep} is missing")
            all_available = False
    
    return all_available

def test_simple_generation(infer_func=None, api_class=None):
    """Test simple audio generation"""
    print("\n=== Testing Simple Generation ===")
    
    if not (infer_func or api_class):
        print("❌ No available generation methods")
        return False
    
    test_text = "สวัสดีครับ"
    ref_text = "สวัสดีครับ ผมเป็นผู้ช่วยเสียงภาษาไทย"
    
    # Try function-based approach first
    if infer_func:
        print("Testing inference function...")
        try:
            # Test different parameter combinations for infer_process
            try:
                # Try without output_path (function returns audio data)
                result = infer_func(
                    gen_text=test_text,
                    ref_text=ref_text
                )
                if result is not None:
                    print("✅ Inference function generated audio successfully (basic params)")
                    return True
            except Exception as e:
                print(f"⚠️ Basic inference failed: {e}")
            
            # Try with different parameter names
            try:
                result = infer_func(
                    text=test_text,
                    ref_text=ref_text
                )
                if result is not None:
                    print("✅ Inference function generated audio successfully (alt params)")
                    return True
            except Exception as e:
                print(f"⚠️ Alternative inference failed: {e}")
                
        except Exception as e:
            print(f"❌ Inference function test failed: {e}")
    
    # Try API-based approach
    if api_class:
        print("Testing API class...")
        try:
            model = api_class()
            
            # Create a temporary reference file using gTTS
            temp_ref_file = None
            try:
                import tempfile
                try:
                    from gtts import gTTS
                    temp_ref_obj = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                    temp_ref_file = temp_ref_obj.name
                    temp_ref_obj.close()
                    
                    ref_tts = gTTS(text=ref_text, lang="th")
                    ref_tts.save(temp_ref_file)
                    print(f"   Created temporary reference file: {temp_ref_file}")
                except ImportError:
                    print("   gTTS not available for reference file creation")
                    return False
                
                # Test with ref_file parameter
                try:
                    result = model.infer(
                        gen_text=test_text,
                        ref_text=ref_text,
                        ref_file=temp_ref_file
                    )
                    print("✅ API class generated audio successfully")
                    return True
                except Exception as e:
                    print(f"⚠️ API class failed: {e}")
                    
                    # Try alternative parameter names
                    try:
                        result = model.infer(
                            text=test_text,
                            ref_text=ref_text,
                            ref_file=temp_ref_file
                        )
                        print("✅ API class generated audio successfully (alt params)")
                        return True
                    except Exception as e2:
                        print(f"⚠️ API class alt params failed: {e2}")
                
            finally:
                # Clean up temp file
                if temp_ref_file and os.path.exists(temp_ref_file):
                    try:
                        os.unlink(temp_ref_file)
                        print(f"   Cleaned up temporary file")
                    except:
                        pass
                
        except Exception as e:
            print(f"❌ API class test failed: {e}")
    
    return False

def main():
    """Main test function"""
    print("F5-TTS-THAI Simple Test Script")
    print("=" * 50)
    
    # Test basic import
    if not test_basic_import():
        print("\n❌ F5-TTS-THAI is not installed or not working")
        print("Install with: pip install git+https://github.com/VYNCX/F5-TTS-THAI.git")
        return
    
    # Test dependencies
    if not test_dependencies():
        print("\n❌ Missing required dependencies")
        print("Install with: pip install torch torchaudio soundfile")
        return
    
    # Test specific imports
    infer_func = test_infer_utils()
    api_class = test_api_import()
    cli_available = test_cli_availability()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if infer_func:
        print("✅ Inference function available")
    if api_class:
        print("✅ API class available")
    if cli_available:
        print("✅ CLI available")
    
    if infer_func or api_class:
        print("\n🧪 Testing audio generation...")
        if test_simple_generation(infer_func, api_class):
            print("✅ F5-TTS-THAI is working correctly!")
        else:
            print("⚠️ F5-TTS-THAI imports work but generation failed")
    else:
        print("❌ No working F5-TTS-THAI methods found")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
