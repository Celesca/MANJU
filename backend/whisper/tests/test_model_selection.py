#!/usr/bin/env python3
"""
Test the updated server with model selection capability
"""

import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_manager():
    """Test the model manager"""
    print("🧪 Testing Model Manager")
    print("=" * 40)
    
    try:
        from model_manager import get_model_manager
        
        manager = get_model_manager()
        print("✅ Model manager created")
        
        # Get available models
        models = manager.get_available_models()
        print(f"✅ Found {len(models)} available models:")
        
        for model in models:
            status = "⭐" if model.get('recommended') else "  "
            print(f"   {status} {model['id']}: {model['name']}")
            print(f"      Type: {model['type']}, Tier: {model['performance_tier']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model manager test failed: {e}")
        return False

def test_server_import():
    """Test server imports"""
    print("\n🧪 Testing Server Imports")
    print("=" * 40)
    
    try:
        # Test FastAPI import
        from fastapi import FastAPI
        print("✅ FastAPI imported")
        
        # Test model manager import
        from model_manager import get_model_manager
        print("✅ Model manager imported")
        
        # Test whisper imports
        from whisper.faster_whisper_thai import WhisperConfig
        print("✅ Whisper config imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Server import test failed: {e}")
        return False

def main():
    print("🎬 Multi-agent Call Center - Model Selection Test")
    print("=" * 60)
    
    # Test model manager
    manager_ok = test_model_manager()
    
    # Test server imports
    import_ok = test_server_import()
    
    print("\n📊 Test Results:")
    print("=" * 40)
    print(f"Model Manager: {'✅ PASS' if manager_ok else '❌ FAIL'}")
    print(f"Server Imports: {'✅ PASS' if import_ok else '❌ FAIL'}")
    
    if manager_ok and import_ok:
        print("\n🎉 All tests passed!")
        print("🚀 Ready to start server with model selection")
        print("\n💡 Available models:")
        print("   • biodatlab-faster (faster-whisper, recommended)")
        print("   • pathumma-large (standard whisper, recommended)")
        print("   • large-v3-faster (faster-whisper)")
        print("   • large-v3-standard (standard whisper)")
        print("   • medium-faster (faster-whisper)")
        print("   • medium-standard (standard whisper)")
        
        print("\n🌐 New API endpoints:")
        print("   GET  /api/models - List available models")
        print("   POST /api/models/{model_id}/load - Load specific model")
        print("   POST /api/asr - Transcribe with model selection")
        print("   POST /api/asr/batch - Batch transcription with model selection")
        
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    return manager_ok and import_ok

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
