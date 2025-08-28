#!/usr/bin/env python3
"""
Simplified test for faster-whisper Thai ASR with Hugging Face model
"""

import os
import sys

def test_basic_import():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    
    try:
        # Test faster_whisper
        import faster_whisper
        print("✅ faster_whisper imported successfully")
        
        # Test our module
        from whisper.faster_whisper_thai import WhisperConfig
        print("✅ WhisperConfig imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_config():
    """Test model configuration"""
    print("\n📋 Testing model configuration...")
    
    try:
        from whisper.faster_whisper_thai import WhisperConfig
        
        config = WhisperConfig()
        print(f"✅ Default config created")
        print(f"   Model: {config.model_name}")
        print(f"   Language: {config.language}")
        print(f"   Device: {config.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def main():
    print("🎬 Quick Thai ASR Test")
    print("=" * 40)
    
    if test_basic_import() and test_model_config():
        print("\n🎉 Basic tests passed!")
        print("💡 Ready to test with audio files")
        return True
    else:
        print("\n❌ Tests failed")
        return False

if __name__ == "__main__":
    main()
