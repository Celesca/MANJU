"""
Quick fix for PyTorch DLL issues on Windows
Run this if you're getting 'shm.dll' or similar errors
"""

import subprocess
import sys
import os

def fix_pytorch_installation():
    """Fix common PyTorch DLL issues on Windows"""
    
    print("🔧 PyTorch & Dependencies Fix Script")
    print("=" * 40)
    
    # Step 1: Clean install
    print("\n1. Cleaning existing installations...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "tensorflow", "keras", "tf-keras", "-y"])
    
    # Step 2: Clear pip cache
    print("\n2. Clearing pip cache...")
    subprocess.run([sys.executable, "-m", "pip", "cache", "purge"])
    
    # Step 3: Install tf-keras for compatibility
    print("\n3. Installing tf-keras for Transformers compatibility...")
    subprocess.run([sys.executable, "-m", "pip", "install", "tf-keras"])
    
    # Step 4: Install CPU-only version (more stable)
    print("\n4. Installing CPU-only PyTorch (recommended for stability)...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], check=True)
        print("✅ PyTorch installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Installation failed. Trying alternative method...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
    
    # Step 4: Test installation
    print("\n5. Testing PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully!")
        print(f"CPU available: {torch.cuda.device_count() == 0 or 'CPU'}")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        print("\n💡 Additional steps needed:")
        print("1. Download and install Visual C++ Redistributable:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("2. Restart your computer")
        print("3. Run this script again")
        return False
    
    return True

if __name__ == "__main__":
    if fix_pytorch_installation():
        print("\n🎉 Fix completed successfully!")
        print("You can now run: streamlit run voice_chatbot.py")
    else:
        print("\n⚠️ Manual intervention required. See instructions above.")
