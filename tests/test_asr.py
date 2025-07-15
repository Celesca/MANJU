"""
Demo script to test the ASR pipeline
"""
from whisper import OverlappingASRPipeline, AudioConfig, ProcessingConfig
import time

def test_asr():
    """Test the ASR pipeline with a sample audio file"""
    
    print("🎤 Testing ASR Pipeline...")
    
    # Configuration
    audio_config = AudioConfig(
        chunk_length_ms=27000,
        overlap_ms=2000,
        sample_rate=16000
    )
    
    processing_config = ProcessingConfig(
        model_name="nectec/Pathumma-whisper-th-large-v3",
        batch_size=2,
        use_gpu=True
    )
    
    # Test with a dummy file path (replace with actual audio file)
    audio_file = "test_audio.wav"  # Replace with your audio file
    
    try:
        # Create pipeline
        pipeline = OverlappingASRPipeline(
            input_path=audio_file,
            audio_config=audio_config,
            processing_config=processing_config
        )
        
        print(f"📁 Processing: {audio_file}")
        
        start_time = time.time()
        
        # Test callable interface
        transcription = pipeline()
        
        end_time = time.time()
        
        print(f"✅ Transcription: {transcription}")
        print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
        
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        print("Please place an audio file in the directory and update the path")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_asr()
