"""
F5-TTS-THAI Model Setup Script
This script helps download and set up the VIZINTZOR Thai model for F5-TTS
"""

import os
import requests
from pathlib import Path

def download_file(url, local_path, description="file"):
    """Download a file with progress indication"""
    print(f"📥 Downloading {description}...")
    print(f"URL: {url}")
    print(f"Saving to: {local_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r⏳ Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
        
        print(f"\n✅ Downloaded {description} successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {description}: {str(e)}")
        return False

def setup_thai_model():
    """Set up the Thai F5-TTS model"""
    print("🎯 F5-TTS-THAI Model Setup")
    print("=" * 50)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model files to download
    base_url = "https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main"
    files_to_download = [
        {
            "filename": "model_1000000.pt",
            "url": f"{base_url}/model_1000000.pt",
            "description": "Thai F5-TTS Model (1.35 GB)",
            "required": True
        },
        {
            "filename": "vocab.txt",
            "url": f"{base_url}/vocab.txt",
            "description": "Thai Vocabulary File (REQUIRED - custom 2587 tokens)",
            "required": True  # This is required for VIZINTZOR model
        },
        {
            "filename": "config.json",
            "url": f"{base_url}/config.json",
            "description": "Model Configuration",
            "required": False
        }
    ]
    
    success_count = 0
    
    for file_info in files_to_download:
        local_path = models_dir / file_info["filename"]
        
        # Check if file already exists
        if local_path.exists():
            print(f"✅ {file_info['filename']} already exists")
            success_count += 1
            continue
        
        # Download the file
        if download_file(file_info["url"], local_path, file_info["description"]):
            success_count += 1
        elif file_info["required"]:
            print(f"❌ Required file {file_info['filename']} failed to download!")
            return False
    
    print("\n" + "=" * 50)
    print(f"✅ Setup complete! Downloaded {success_count}/{len(files_to_download)} files")
    
    # Provide usage instructions
    print("\n📋 Usage Instructions:")
    print("1. The Thai model is now available in the 'models/' directory")
    print("2. Run your voice chatbot: streamlit run voice_chatbot.py")
    print("3. Select 'F5-TTS-THAI' as your TTS engine in the sidebar")
    print("4. The system will automatically use the Thai model for authentic pronunciation")
    
    # Check model file size
    model_path = models_dir / "model_1000000.pt"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\n📏 Model size: {size_mb:.1f} MB")
        if size_mb < 100:
            print("⚠️  Warning: Model file seems too small. Download may be incomplete.")
        else:
            print("✅ Model file size looks correct!")
    
    return True

def verify_installation():
    """Verify that F5-TTS-THAI is properly installed"""
    print("\n🔍 Verifying F5-TTS-THAI Installation...")
    
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        print("❌ PyTorch not found. Install with: pip install torch torchaudio")
        return False
    
    try:
        import soundfile as sf
        print("✅ SoundFile available")
    except ImportError:
        print("❌ SoundFile not found. Install with: pip install soundfile")
        return False
    
    try:
        # Try importing F5-TTS
        from f5_tts.api import F5TTS
        print("✅ F5-TTS API available")
    except ImportError:
        try:
            from f5_tts.infer.utils_infer import infer_process
            print("✅ F5-TTS inference function available")
            
            # Print function signature for debugging
            try:
                import inspect
                sig = inspect.signature(infer_process)
                print(f"📋 infer_process signature: {sig}")
            except:
                pass
                
        except ImportError:
            print("❌ F5-TTS not found. Install with:")
            print("   pip install git+https://github.com/VYNCX/F5-TTS-THAI.git")
            return False
    
    # Check for model files
    model_path = Path("models/model_1000000.pt")
    if model_path.exists():
        print(f"✅ Thai model found: {model_path}")
    else:
        print(f"❌ Thai model not found: {model_path}")
        return False
    
    print("✅ All dependencies verified!")
    return True

if __name__ == "__main__":
    print("🚀 F5-TTS-THAI Setup Script")
    print("This script will download the VIZINTZOR Thai model for authentic Thai TTS")
    
    # Verify installation first
    if not verify_installation():
        print("\n❌ Please install missing dependencies first")
        print("\nInstallation commands:")
        print("pip install torch torchaudio soundfile")
        print("pip install git+https://github.com/VYNCX/F5-TTS-THAI.git")
        exit(1)
    
    # Ask user for confirmation
    print(f"\n⚠️  Warning: This will download ~1.4 GB of model files")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        if setup_thai_model():
            print("\n🎉 Setup completed successfully!")
            print("You can now use authentic Thai TTS in your voice chatbot!")
        else:
            print("\n❌ Setup failed. Please try again or download manually:")
            print("https://huggingface.co/VIZINTZOR/F5-TTS-THAI")
    else:
        print("Setup cancelled.")
        print("\nTo download manually:")
        print("1. Go to: https://huggingface.co/VIZINTZOR/F5-TTS-THAI")
        print("2. Download 'model_1000000.pt' to a 'models/' folder")
        print("3. Optionally download 'vocab.txt' and 'config.json'")
