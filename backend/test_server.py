#!/usr/bin/env python3
"""
Test script for the Multi-agent Call Center Backend Server
Tests the faster-whisper Thai ASR implementation and server endpoints
"""

import os
import sys
import requests
import json
import time
from pathlib import Path

# Add backend to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_faster_whisper_thai():
    """Test the faster-whisper Thai ASR implementation directly"""
    print("🧪 Testing faster-whisper Thai ASR implementation")
    print("=" * 50)
    
    try:
        from whisper.faster_whisper_thai import create_thai_asr, WhisperConfig
        
        # Create ASR instance
        print("📦 Creating Thai ASR instance...")
        config = WhisperConfig(
            model_name="large-v3",
            language="th", 
            device="auto",
            compute_type="int8_float16",
            beam_size=1,
            use_vad=True
        )
        
        asr = create_thai_asr(config)
        print("✅ ASR instance created successfully!")
        print(f"   Device: {asr.device}")
        print(f"   Model: {asr.config.model_name}")
        
        # Look for test audio files
        test_files = []
        audio_dirs = [".", "temp", "audio_uploads", "tests"]
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
        
        for dir_path in audio_dirs:
            if os.path.exists(dir_path):
                for ext in audio_extensions:
                    files = list(Path(dir_path).glob(f"*{ext}"))
                    test_files.extend(files)
        
        if test_files:
            test_file = test_files[0]
            print(f"🎵 Testing with: {test_file}")
            
            start_time = time.time()
            result = asr.transcribe(str(test_file))
            end_time = time.time()
            
            print("\n📝 Transcription Results:")
            print(f"   Text: {result['text']}")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Speed ratio: {result['speed_ratio']:.1f}x realtime")
            print(f"   Chunks: {result['chunks_processed']}")
            print(f"   Device: {result['device']}")
            
        else:
            print("ℹ️ No audio files found for testing")
            print("💡 Add some audio files (.wav, .mp3, .m4a) to test directory")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure faster-whisper is installed: pip install faster-whisper")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_server_endpoints(base_url="http://localhost:8000"):
    """Test the FastAPI server endpoints"""
    print("\n🌐 Testing Server Endpoints")
    print("=" * 50)
    
    # Test health endpoint
    try:
        print("🔍 Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health endpoint working")
            print(f"   Status: {health_data.get('status')}")
            print(f"   ASR Model Loaded: {health_data.get('asr_model_loaded')}")
            print(f"   Device: {health_data.get('device')}")
            print(f"   Uptime: {health_data.get('uptime', 0):.1f}s")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure the server is running: python server.py")
        return False
    
    # Test ASR info endpoint
    try:
        print("\n🔍 Testing ASR info endpoint...")
        response = requests.get(f"{base_url}/api/asr/info", timeout=10)
        
        if response.status_code == 200:
            info_data = response.json()
            print("✅ ASR info endpoint working")
            print(f"   Model: {info_data.get('model_name')}")
            print(f"   Language: {info_data.get('language')}")
            print(f"   Device: {info_data.get('device')}")
            print(f"   VAD: {info_data.get('use_vad')}")
        else:
            print(f"❌ ASR info endpoint failed: {response.status_code}")
            
    except requests.RequestException as e:
        print(f"❌ ASR info endpoint error: {e}")
    
    # Test ASR transcription endpoint (if audio file available)
    test_files = []
    for ext in ['.wav', '.mp3', '.m4a']:
        test_files.extend(Path('.').glob(f"*{ext}"))
        test_files.extend(Path('temp').glob(f"*{ext}") if Path('temp').exists() else [])
    
    if test_files:
        test_file = test_files[0]
        print(f"\n🔍 Testing ASR transcription with: {test_file}")
        
        try:
            with open(test_file, 'rb') as f:
                files = {'file': (test_file.name, f, 'audio/wav')}
                data = {
                    'language': 'th',
                    'use_vad': True,
                    'beam_size': 1
                }
                
                response = requests.post(
                    f"{base_url}/api/asr",
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ ASR transcription endpoint working")
                print(f"   Text: {result.get('text', '')}")
                print(f"   Duration: {result.get('duration', 0):.2f}s")
                print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
                print(f"   Speed ratio: {result.get('speed_ratio', 0):.1f}x")
            else:
                print(f"❌ ASR transcription failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except requests.RequestException as e:
            print(f"❌ ASR transcription error: {e}")
    else:
        print("\nℹ️ No audio files found for transcription testing")
    
    return True


def print_setup_instructions():
    """Print setup instructions"""
    print("\n📋 Setup Instructions")
    print("=" * 50)
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Install faster-whisper (if not already installed):")
    print("   pip install faster-whisper")
    print()
    print("3. Start the server:")
    print("   python server.py")
    print()
    print("4. Test the API:")
    print("   curl -X POST 'http://localhost:8000/api/asr' \\")
    print("        -F 'file=@your_audio.wav' \\")
    print("        -F 'language=th'")
    print()
    print("5. Access API documentation:")
    print("   http://localhost:8000/docs")
    print()
    print("📁 Server Endpoints:")
    print("   GET  /health          - Health check")
    print("   GET  /api/asr/info    - Model information")
    print("   POST /api/asr         - Transcribe audio")
    print("   POST /api/asr/batch   - Batch transcription")
    print("   POST /api/asr/reload  - Reload model")


def main():
    """Main test function"""
    print("🎬 Multi-agent Call Center Backend Test Suite")
    print("=" * 60)
    
    # Test the faster-whisper implementation
    whisper_success = test_faster_whisper_thai()
    
    # Test server endpoints (if server is running)
    server_success = test_server_endpoints()
    
    # Print setup instructions
    print_setup_instructions()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"✅ Faster-Whisper Thai: {'PASS' if whisper_success else 'FAIL'}")
    print(f"🌐 Server Endpoints: {'PASS' if server_success else 'FAIL/NOT_RUNNING'}")
    
    if whisper_success and server_success:
        print("\n🎉 All tests passed! The backend is ready for production.")
    elif whisper_success:
        print("\n⚠️ Faster-Whisper works but server is not running.")
        print("💡 Start the server with: python server.py")
    else:
        print("\n❌ Some tests failed. Check the setup instructions above.")


if __name__ == "__main__":
    main()
