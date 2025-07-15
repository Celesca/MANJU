#!/usr/bin/env python3
"""
Simple script to download the VIZINTZOR vocab file
"""

import os
import requests
from pathlib import Path

def download_vocab():
    """Download the vocab file from VIZINTZOR repository"""
    print("📥 Downloading VIZINTZOR vocab.txt...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download vocab file
    vocab_url = "https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main/vocab.txt"
    vocab_path = models_dir / "vocab.txt"
    
    try:
        response = requests.get(vocab_url)
        response.raise_for_status()
        
        with open(vocab_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded vocab to: {vocab_path}")
        print(f"📏 Size: {vocab_path.stat().st_size} bytes")
        
        # Check content briefly
        with open(vocab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"📊 Vocab contains {len(lines)} tokens")
            print(f"🔤 First few tokens: {lines[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    download_vocab()
