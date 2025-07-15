#!/usr/bin/env python3
"""
F5-TTS-THAI Diagnostic Test Script
This script tests various F5-TTS-THAI installation and usage methods.
"""

import sys
import subprocess
import traceback

def test_import():
    """Test importing F5-TTS"""
    print("🧪 Testing F5-TTS imports...")
    
    # Test basic import
    try:
        import f5_tts
        print("✅ f5_tts module imported successfully")
        print(f"   Module path: {f5_tts.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import f5_tts: {e}")
        return False
    
    # Test specific imports
    imports_to_test = [
        ("f5_tts.api", "F5TTS"),
        ("f5_tts.model", "F5TTS"),
        ("f5_tts.infer.utils_infer", "infer_process"),
        ("f5_tts.infer.infer_cli", None)
    ]
    
    for module_name, class_name in imports_to_test:
        try:
            if class_name:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                print(f"✅ {module_name}.{class_name} imported successfully")
            else:
                __import__(module_name)
                print(f"✅ {module_name} imported successfully")
        except ImportError as e:
            print(f"⚠️  {module_name}.{class_name if class_name else ''} import failed: {e}")
    
    return True

def test_cli():
    """Test F5-TTS CLI"""
    print("\n🖥️  Testing F5-TTS CLI...")
    
    try:
        # Test help command to see available parameters
        result = subprocess.run(
            ['python', '-m', 'f5_tts.infer.infer_cli', '--help'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0:
            print("✅ F5-TTS CLI is accessible")
            print("📋 Available CLI parameters:")
            # Extract parameter information
            lines = result.stdout.split('\n')
            for line in lines:
                if line.strip().startswith('-'):
                    print(f"   {line.strip()}")
        else:
            print(f"❌ F5-TTS CLI failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")

def test_api_creation():
    """Test creating F5TTS API object"""
    print("\n🔧 Testing F5TTS API object creation...")
    
    try:
        # Try different import paths
        api_classes = [
            ("f5_tts.api", "F5TTS"),
            ("f5_tts.model", "F5TTS"),
            ("f5_tts", "F5TTS")
        ]
        
        for module_name, class_name in api_classes:
            try:
                module = __import__(module_name, fromlist=[class_name])
                F5TTSClass = getattr(module, class_name)
                
                # Try to create instance
                model = F5TTSClass(model_type="F5-TTS")
                print(f"✅ Successfully created F5TTS instance from {module_name}.{class_name}")
                
                # Try to get the infer method signature
                if hasattr(model, 'infer'):
                    import inspect
                    sig = inspect.signature(model.infer)
                    print(f"   infer() parameters: {list(sig.parameters.keys())}")
                    return True
                else:
                    print(f"⚠️  Model has no 'infer' method")
                    
            except Exception as e:
                print(f"⚠️  Failed to create from {module_name}.{class_name}: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ API creation test failed: {e}")
        return False

def test_inference_function():
    """Test inference function approach"""
    print("\n⚙️ Testing inference function...")
    
    try:
        from f5_tts.infer.utils_infer import infer_process
        
        import inspect
        sig = inspect.signature(infer_process)
        print(f"✅ infer_process function available")
        print(f"   Parameters: {list(sig.parameters.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"❌ infer_process import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Inference function test failed: {e}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n📦 Testing dependencies...")
    
    deps = [
        'torch',
        'torchaudio', 
        'soundfile',
        'numpy'
    ]
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} missing - install with: pip install {dep}")

def test_minimal_generation():
    """Test minimal audio generation"""
    print("\n🎵 Testing minimal audio generation...")
    
    # Only proceed if basic imports work
    try:
        import f5_tts
    except ImportError:
        print("❌ Skipping generation test - F5-TTS not installed")
        return
    
    test_text = "สวัสดีครับ"
    
    print(f"   Testing with text: '{test_text}'")
    
    # Test CLI approach with correct parameters
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            output_file = temp_file.name
        
        cmd = [
            'python', '-m', 'f5_tts.infer.infer_cli',
            '--gen_text', test_text,
            '--output_file', output_file,
            '--model', 'F5-TTS'
        ]
        
        print(f"   Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ Audio generated successfully! File size: {file_size} bytes")
            
            # Clean up
            try:
                os.unlink(output_file)
            except:
                pass
                
        else:
            print(f"❌ Generation failed:")
            print(f"   Return code: {result.returncode}")
            print(f"   Stdout: {result.stdout}")
            print(f"   Stderr: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        traceback.print_exc()

def main():
    print("🔍 F5-TTS-THAI Diagnostic Test")
    print("=" * 50)
    
    # Run all tests
    test_dependencies()
    
    if test_import():
        test_cli()
        test_api_creation()
        test_inference_function()
        test_minimal_generation()
    else:
        print("\n❌ Basic import failed. Please install F5-TTS-THAI:")
        print("   pip install torch torchaudio")
        print("   pip install git+https://github.com/VYNCX/F5-TTS-THAI.git")
    
    print("\n" + "=" * 50)
    print("🏁 Diagnostic complete!")

if __name__ == "__main__":
    main()
