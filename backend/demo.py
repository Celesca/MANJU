#!/usr/bin/env python3
"""
Demo script for the Multi-agent Call Center Backend
Demonstrates the faster-whisper Thai ASR functionality
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add backend to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_demo_audio():
    """Create a simple demo audio file for testing (if none exists)"""
    print("🎵 Looking for demo audio files...")
    
    # Look for existing audio files
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
    audio_dirs = ['.', 'temp', 'audio_uploads', 'tests']
    
    demo_files = []
    for dir_path in audio_dirs:
        if os.path.exists(dir_path):
            for ext in audio_extensions:
                files = list(Path(dir_path).glob(f"*{ext}"))
                demo_files.extend(files)
    
    if demo_files:
        print(f"✅ Found {len(demo_files)} audio files for demo")
        return str(demo_files[0])
    
    print("ℹ️ No audio files found for demo")
    print("💡 Please add some audio files (.wav, .mp3, .m4a) to test the system")
    return None


def demo_faster_whisper():
    """Demonstrate the faster-whisper Thai ASR"""
    print("🎬 Multi-agent Call Center Backend Demo")
    print("=" * 60)
    
    try:
        from whisper.faster_whisper_thai import create_thai_asr, WhisperConfig
        
        print("📦 Initializing Thai ASR with faster-whisper...")
        
        # Create optimized configuration
        config = WhisperConfig(
            model_name="large-v3",
            language="th",
            device="auto",
            compute_type="int8_float16",
            beam_size=1,
            use_vad=True,
            chunk_length_ms=30000,
            overlap_ms=1000
        )
        
        # Create ASR instance
        start_time = time.time()
        asr = create_thai_asr(config)
        init_time = time.time() - start_time
        
        print(f"✅ ASR initialized in {init_time:.2f}s")
        print(f"   Device: {asr.device}")
        print(f"   Model: {asr.config.model_name}")
        print(f"   Language: {asr.config.language}")
        print(f"   VAD: {'Enabled' if asr.config.use_vad else 'Disabled'}")
        
        # Find demo audio
        demo_file = create_demo_audio()
        
        if demo_file:
            print(f"\n🎵 Testing with: {Path(demo_file).name}")
            print("   Processing audio...")
            
            # Transcribe audio
            start_time = time.time()
            result = asr.transcribe(demo_file)
            total_time = time.time() - start_time
            
            # Display results
            print("\n📝 Transcription Results:")
            print("=" * 50)
            print(f"Text: {result['text']}")
            print(f"Language: {result['language']}")
            print(f"Audio duration: {result['duration']:.2f}s")
            print(f"Processing time: {result['processing_time']:.2f}s")
            print(f"Total time: {total_time:.2f}s")
            print(f"Speed ratio: {result['speed_ratio']:.1f}x realtime")
            print(f"Chunks processed: {result['chunks_processed']}")
            print(f"Model: {result['model']}")
            print(f"Device: {result['device']}")
            
            # Performance analysis
            print("\n📊 Performance Analysis:")
            print("=" * 50)
            if result['speed_ratio'] > 1.0:
                print(f"🚀 Excellent! Processing {result['speed_ratio']:.1f}x faster than realtime")
            elif result['speed_ratio'] > 0.5:
                print(f"✅ Good performance at {result['speed_ratio']:.1f}x realtime")
            else:
                print(f"⚠️ Slow processing at {result['speed_ratio']:.1f}x realtime")
                print("💡 Consider using GPU or reducing beam_size for better performance")
            
            # Quality indicators
            if len(result['text']) > 0:
                print("✅ Transcription successful")
                if len(result['text']) > 50:
                    print("✅ Good text length - likely accurate transcription")
            else:
                print("⚠️ Empty transcription - check audio quality or VAD settings")
        
        else:
            print("\n📋 Demo Summary (No audio files):")
            print("=" * 50)
            print("✅ faster-whisper Thai ASR model loaded successfully")
            print("✅ Configuration optimized for Thai language")
            print("✅ Ready to process audio files")
            print("\n💡 To test transcription:")
            print("   1. Add some Thai audio files (.wav, .mp3, .m4a)")
            print("   2. Run this demo again")
            print("   3. Or start the server: python server.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Installation Required:")
        print("   pip install faster-whisper")
        print("   pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def demo_server_info():
    """Show server information and usage examples"""
    print("\n🌐 Server Information")
    print("=" * 60)
    print("Start the server with:")
    print("   python server.py")
    print("   # or")
    print("   start_server.bat")
    print()
    print("Server endpoints:")
    print("   🌐 API Base: http://localhost:8000")
    print("   📚 Docs: http://localhost:8000/docs")
    print("   🔍 Health: http://localhost:8000/health")
    print("   📊 ASR Info: http://localhost:8000/api/asr/info")
    print("   🎵 Transcribe: POST http://localhost:8000/api/asr")
    print()
    print("API Usage Examples:")
    print("   # Health check")
    print("   curl http://localhost:8000/health")
    print()
    print("   # Transcribe audio")
    print("   curl -X POST 'http://localhost:8000/api/asr' \\")
    print("        -F 'file=@audio.wav' \\")
    print("        -F 'language=th'")
    print()
    print("   # Python client")
    print("   python api_client.py --transcribe audio.wav")
    print()
    print("Test the system:")
    print("   python test_server.py")


def main():
    """Main demo function"""
    print("🎭 Multi-agent Call Center Backend")
    print("🚀 Thai ASR with faster-whisper Demo")
    print("=" * 60)
    
    # Test faster-whisper implementation
    whisper_success = demo_faster_whisper()
    
    # Show server information
    demo_server_info()
    
    # Final summary
    print("\n🎯 Next Steps:")
    print("=" * 60)
    
    if whisper_success:
        print("✅ faster-whisper Thai ASR is working!")
        print("1. Start the server: python server.py")
        print("2. Test the API: python api_client.py --health")
        print("3. Upload audio files via API or web interface")
        print("4. Check the documentation: http://localhost:8000/docs")
    else:
        print("❌ Setup incomplete. Please:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install faster-whisper: pip install faster-whisper")
        print("3. Run demo again: python demo.py")
    
    print("\n📚 For more information, see README.md")


if __name__ == "__main__":
    main()
