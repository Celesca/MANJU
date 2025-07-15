#!/usr/bin/env python3
"""
Installation script for F5-TTS-THAI dependencies and model
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(command) if isinstance(command, list) else command}")
    
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        if result.stdout:
            print(f"✅ Success: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def install_dependencies():
    """Install required dependencies for F5-TTS-THAI"""
    print("📦 Installing F5-TTS-THAI dependencies...")
    
    # Core dependencies
    dependencies = [
        "torch",
        "torchaudio", 
        "soundfile",
        "huggingface_hub",
        "accelerate",
        "transformers",
        "pydub",
        "librosa",
        "numpy",
        "scipy"
    ]
    
    print("Installing core dependencies...")
    for dep in dependencies:
        success = run_command([sys.executable, "-m", "pip", "install", dep], 
                            f"Installing {dep}")
        if not success:
            print(f"⚠️ Warning: Failed to install {dep}")
    
    # Install F5-TTS from source (VIZINTZOR version)
    print("\n🔧 Installing F5-TTS-THAI from source...")
    f5_tts_commands = [
        "git clone https://github.com/VYNCX/F5-TTS-THAI.git",
        "cd F5-TTS-THAI && pip install -e ."
    ]
    
    for cmd in f5_tts_commands:
        success = run_command(cmd, f"Running: {cmd}")
        if not success:
            print(f"⚠️ Warning: Command failed: {cmd}")
    
    print("\n✅ F5-TTS-THAI installation completed!")

def download_thai_model():
    """Download the Thai model checkpoint"""
    print("\n📥 Downloading VIZINTZOR Thai model...")
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Download model checkpoint
        model_path = hf_hub_download(
            repo_id="VIZINTZOR/F5-TTS-THAI",
            filename="model_1000000.pt",
            local_dir="./models"
        )
        print(f"✅ Model downloaded to: {model_path}")
        
        # Download vocab file
        try:
            vocab_path = hf_hub_download(
                repo_id="VIZINTZOR/F5-TTS-THAI",
                filename="vocab.txt",
                local_dir="./models"
            )
            print(f"✅ Vocab downloaded to: {vocab_path}")
        except Exception as e:
            print(f"⚠️ Vocab download failed (optional): {e}")
        
        return True
        
    except ImportError:
        print("❌ huggingface_hub not available. Installing...")
        success = run_command([sys.executable, "-m", "pip", "install", "huggingface_hub"], 
                            "Installing huggingface_hub")
        if success:
            return download_thai_model()  # Retry
        return False
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

def verify_installation():
    """Verify the installation"""
    print("\n🔍 Verifying installation...")
    
    try:
        # Test imports
        import torch
        print("✅ PyTorch available")
        
        import torchaudio
        print("✅ TorchAudio available")
        
        import soundfile as sf
        print("✅ SoundFile available")
        
        try:
            from f5_tts.model import DiT
            from f5_tts.infer.utils_infer import load_model
            print("✅ F5-TTS core components available")
        except ImportError as e:
            print(f"⚠️ F5-TTS import issue: {e}")
        
        # Check model file
        model_paths = [
            "models/model_1000000.pt",
            "model_1000000.pt",
            "F5-TTS-THAI/model_1000000.pt"
        ]
        
        model_found = False
        for path in model_paths:
            if os.path.exists(path):
                print(f"✅ Thai model found: {path}")
                model_found = True
                break
        
        if not model_found:
            print("⚠️ Thai model not found. Run download_thai_model() or check setup_thai_model.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main installation process"""
    print("🚀 F5-TTS-THAI Installation Script")
    print("=" * 50)
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2: Download model
    download_thai_model()
    
    # Step 3: Verify installation
    verify_installation()
    
    print("\n🎉 Installation complete!")
    print("\nNext steps:")
    print("1. Run: python test_thai_model.py")
    print("2. If tests pass, run: streamlit run voice_chatbot.py")
    print("3. Check F5_TTS_THAI_SETUP.md for troubleshooting")

if __name__ == "__main__":
    main()
