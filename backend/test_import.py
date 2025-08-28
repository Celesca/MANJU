#!/usr/bin/env python3
"""
Simple test to verify the faster-whisper Thai implementation works
Tests the import and basic functionality
"""

import os
import sys

def test_import():
    """Test if we can import the faster-whisper Thai module"""
    print("🧪 Testing faster-whisper Thai import...")
    
    try:
        # Try direct import from current directory
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'whisper'))
        
        from faster_whisper_thai import FasterWhisperThai, WhisperConfig, create_thai_asr
        print("✅ Import successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_creation():
    """Test creating the ASR model"""
    print("\n📦 Testing model creation...")
    
    try:
        from faster_whisper_thai import WhisperConfig, create_thai_asr
        
        # Create config with the Thai model
        config = WhisperConfig(
            model_name="Vinxscribe/biodatlab-whisper-th-large-v3-faster",
            language="th",
            device="auto"
        )
        
        print(f"   Using model: {config.model_name}")
        print(f"   Language: {config.language}")
        print(f"   Device: {config.device}")
        
        # Try to create ASR instance (this will download the model if needed)
        asr = create_thai_asr(config)
        print("✅ Model creation successful!")
        print(f"   Device used: {asr.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        if "faster-whisper" in str(e):
            print("💡 Install faster-whisper: pip install faster-whisper")
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    print("\n🔍 Checking dependencies...")
    
    dependencies = [
        ("faster_whisper", "faster-whisper"),
        ("pydub", "pydub"),
        ("numpy", "numpy"),
        ("pathlib", "pathlib (built-in)"),
    ]
    
    missing = []
    
    for module, package in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n📥 Install missing dependencies:")
        for package in missing:
            if package != "pathlib (built-in)":
                print(f"   pip install {package}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🎬 faster-whisper Thai ASR Test")
    print("=" * 50)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    if not deps_ok:
        print("\n❌ Missing dependencies. Install them first.")
        return False
    
    # Test import
    import_ok = test_import()
    
    if not import_ok:
        print("\n❌ Import failed. Check the module structure.")
        return False
    
    # Test model creation
    model_ok = test_model_creation()
    
    if model_ok:
        print("\n🎉 All tests passed!")
        print("✅ faster-whisper Thai ASR is ready to use")
        print("🚀 You can now start the server: python server.py")
    else:
        print("\n⚠️ Model creation failed, but imports work")
        print("💡 This might be due to network issues or missing dependencies")
    
    return model_ok

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
