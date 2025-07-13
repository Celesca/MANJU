"""
Python setup script for Voice Chatbot
This script handles PyTorch installation issues on Windows
"""

import subprocess
import sys
import os
import importlib.util

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - Timeout")
        return False
    except Exception as e:
        print(f"💥 {description} - Exception: {e}")
        return False

def check_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_module(module_name):
    """Check if a Python module is installed and importable"""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            importlib.import_module(module_name)
            return True
    except:
        pass
    return False

def main():
    print("🎤 Voice Chatbot Setup Script")
    print("=" * 50)
    
    # Step 1: Uninstall existing PyTorch
    print("\n📦 Step 1: Cleaning PyTorch installation...")
    run_command("pip uninstall torch torchvision torchaudio -y", "Uninstalling existing PyTorch")
    
    # Step 2: Install PyTorch based on GPU support
    print("\n🎯 Step 2: Installing PyTorch...")
    has_gpu = check_gpu()
    
    if has_gpu:
        print("🚀 NVIDIA GPU detected! Installing CUDA-enabled PyTorch...")
        pytorch_success = run_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "Installing PyTorch with CUDA"
        )
    else:
        print("💻 No GPU detected. Installing CPU-only PyTorch...")
        pytorch_success = run_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            "Installing PyTorch CPU-only"
        )
    
    # Fallback PyTorch installation
    if not pytorch_success:
        print("🔄 Trying alternative PyTorch installation...")
        pytorch_success = run_command("pip install torch torchvision torchaudio", "Installing PyTorch (fallback)")
    
    # Step 3: Install other requirements
    print("\n📚 Step 3: Installing other dependencies...")
    run_command("pip install -r requirements.txt", "Installing requirements.txt")
    
    # Step 4: Test PyTorch
    print("\n🧪 Step 4: Testing PyTorch installation...")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"🎯 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔥 CUDA version: {torch.version.cuda}")
            print(f"💾 GPU count: {torch.cuda.device_count()}")
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        print("💡 Suggestion: Install Visual C++ Redistributable from:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    
    # Step 5: Test other imports
    print("\n🔍 Step 5: Testing other imports...")
    modules_to_test = ['transformers', 'streamlit', 'pydub', 'requests']
    
    for module in modules_to_test:
        if check_module(module):
            print(f"✅ {module} - OK")
        else:
            print(f"❌ {module} - Failed")
            run_command(f"pip install {module}", f"Installing {module}")
    
    # Step 6: Check Ollama
    print("\n🧠 Step 6: Checking Ollama...")
    if run_command("ollama --version", "Checking Ollama version"):
        run_command("ollama pull phi3", "Downloading Phi3 model")
    else:
        print("⚠️  Ollama not found. Please install from: https://ollama.ai")
        print("   After installation, run: ollama pull phi3")
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. If you see DLL errors, install Visual C++ Redistributable:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("2. Start Ollama server: ollama serve")
    print("3. Run the chatbot: streamlit run voice_chatbot.py")
    print("\n🔧 Troubleshooting:")
    print("- If PyTorch fails to import, restart your computer after installing VC++ Redistributable")
    print("- Check Windows Defender/Antivirus isn't blocking the installation")
    print("- Try running as Administrator if needed")

if __name__ == "__main__":
    main()
