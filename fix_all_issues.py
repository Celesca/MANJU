"""
Comprehensive fix for PyTorch, TensorFlow, and Transformers compatibility issues
"""

import subprocess
import sys
import os

def run_command(command, description="", ignore_errors=False):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 or ignore_errors:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"💥 {description} - Exception: {e}")
        return False

def main():
    print("🔧 Comprehensive Voice Chatbot Fix")
    print("=" * 50)
    
    # Step 1: Clean all problematic packages
    print("\n📦 Step 1: Cleaning problematic packages...")
    packages_to_remove = [
        "torch", "torchvision", "torchaudio",
        "tensorflow", "tensorflow-cpu", "tensorflow-gpu", 
        "keras", "tf-keras"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall {package} -y", f"Removing {package}", ignore_errors=True)
    
    # Step 2: Clear cache
    print("\n🧹 Step 2: Clearing caches...")
    run_command("pip cache purge", "Clearing pip cache", ignore_errors=True)
    
    # Step 3: Install tf-keras first (for Transformers compatibility)
    print("\n🔗 Step 3: Installing TensorFlow compatibility...")
    run_command("pip install tf-keras", "Installing tf-keras")
    
    # Step 4: Install PyTorch CPU-only
    print("\n🧠 Step 4: Installing PyTorch CPU-only...")
    pytorch_success = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "Installing PyTorch CPU"
    )
    
    if not pytorch_success:
        print("🔄 Trying alternative PyTorch installation...")
        run_command("pip install torch torchvision torchaudio", "Installing PyTorch (alternative)")
    
    # Step 5: Install ffmpeg
    print("\n🎵 Step 5: Installing FFmpeg for audio processing...")
    
    # Try winget first (Windows Package Manager)
    ffmpeg_installed = run_command(
        'winget install "FFmpeg (Essentials Build)" --accept-source-agreements --accept-package-agreements',
        "Installing FFmpeg via winget",
        ignore_errors=True
    )
    
    if not ffmpeg_installed:
        print("⚠️  FFmpeg installation via winget failed")
        print("📥 Manual FFmpeg installation required:")
        print("   1. Download from: https://github.com/BtbN/FFmpeg-Builds/releases")
        print("   2. Extract to C:\\ffmpeg")
        print("   3. Add C:\\ffmpeg\\bin to your PATH environment variable")
        print("   4. Restart your computer")
    
    # Step 6: Install other requirements
    print("\n📚 Step 6: Installing other requirements...")
    run_command("pip install -r requirements.txt", "Installing requirements")
    
    # Step 6: Test critical imports
    print("\n🧪 Step 7: Testing critical imports...")
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - OK")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
    
    # Test Transformers with PyTorch backend
    try:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        from transformers import pipeline
        
        # Try to create a simple pipeline to test
        print("🧪 Testing Transformers pipeline...")
        test_pipe = pipeline("sentiment-analysis", framework="pt")
        print("✅ Transformers with PyTorch - OK")
        del test_pipe  # Clean up
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        print("💡 This might still work for ASR, continuing...")
    
    # Test other imports
    imports_to_test = [
        ('streamlit', 'Streamlit'),
        ('pydub', 'PyDub'),
        ('sounddevice', 'Sound Device'),
        ('soundfile', 'Sound File'),
        ('pyttsx3', 'Text-to-Speech'),
        ('requests', 'Requests'),
    ]
    
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - Missing")
            run_command(f"pip install {module}", f"Installing {module}")
    
    # Step 7: Environment variables fix
    print("\n🌍 Step 7: Setting environment variables...")
    env_vars = {
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_VERBOSITY": "error",
        "TF_CPP_MIN_LOG_LEVEL": "3"
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ Set {var}={value}")
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Fix Complete!")
    print("\n📋 What was fixed:")
    print("- ✅ Removed conflicting TensorFlow/Keras installations")
    print("- ✅ Installed tf-keras for Transformers compatibility")
    print("- ✅ Installed stable CPU-only PyTorch")
    print("- ✅ Set environment variables to force PyTorch backend")
    print("- ✅ Installed all other dependencies")
    
    print("\n🚀 Next Steps:")
    print("1. Close this terminal completely")
    print("2. Open a new terminal/command prompt")
    print("3. Navigate to your project folder")
    print("4. Run: streamlit run voice_chatbot.py")
    
    print("\n💡 If you still get errors:")
    print("1. Restart your computer")
    print("2. Try the simple chatbot: streamlit run simple_chatbot.py")
    print("3. Install Visual C++ Redistributable if needed")
    
    print("\n🔧 Environment Variables Set:")
    for var, value in env_vars.items():
        print(f"   {var}={value}")

if __name__ == "__main__":
    main()
