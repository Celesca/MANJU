#!/usr/bin/env python3
"""
Quick test of the optimized Whisper ASR with faster-whisper
"""

import os
import time
from whisper import OverlappingASRPipeline, AudioConfig, ProcessingConfig

def test_optimized_asr():
    """Test the optimized ASR pipeline"""
    print("🧪 Testing Optimized Thai ASR with faster-whisper")
    print("=" * 50)
    
    # Find test audio files
    test_files = []
    for ext in ['.wav', '.mp3', '.m4a']:
        files = [f for f in os.listdir('.') if f.endswith(ext)]
        test_files.extend(files)
    
    if not test_files:
        print("❌ No audio files found in current directory")
        print("💡 Add some .wav, .mp3, or .m4a files to test")
        return
    
    test_file = test_files[0]
    print(f"🎵 Testing with: {test_file}")
    
    # Test configurations
    configs = [
        {
            "name": "Standard Whisper (Baseline)",
            "config": ProcessingConfig(
                model_name="nectec/Pathumma-whisper-th-large-v3",
                use_faster_whisper=False,
                use_gpu=True,
                batch_size=4
            )
        },
        {
            "name": "faster-whisper (Optimized)",
            "config": ProcessingConfig(
                use_faster_whisper=True,
                faster_whisper_model="large-v3",
                compute_type="int8_float16",
                beam_size=1,
                use_vad=True,
                use_gpu=True
            )
        }
    ]
    
    # Audio config optimized for speed
    audio_config = AudioConfig(
        chunk_length_ms=30000,
        overlap_ms=1000,
        sample_rate=16000
    )
    
    results = []
    
    for test_config in configs:
        print(f"\n🔄 Testing: {test_config['name']}")
        print("-" * 30)
        
        try:
            # Create pipeline
            pipeline = OverlappingASRPipeline(
                input_path=test_file,
                audio_config=audio_config,
                processing_config=test_config['config']
            )
            
            # Time the transcription
            start_time = time.time()
            transcription = pipeline(test_file)
            end_time = time.time()
            
            elapsed = end_time - start_time
            results.append({
                'name': test_config['name'],
                'time': elapsed,
                'transcription': transcription[:100] + "..." if len(transcription) > 100 else transcription
            })
            
            print(f"⏱️  Time: {elapsed:.2f} seconds")
            print(f"📝 Result: {transcription[:80]}{'...' if len(transcription) > 80 else ''}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'name': test_config['name'],
                'time': None,
                'transcription': f"Error: {e}"
            })
    
    # Compare results
    print("\n📊 Performance Comparison")
    print("=" * 50)
    
    valid_results = [r for r in results if r['time'] is not None]
    
    if len(valid_results) >= 2:
        baseline = valid_results[0]
        optimized = valid_results[1]
        
        speedup = baseline['time'] / optimized['time']
        
        print(f"📈 Results:")
        print(f"   {baseline['name']}: {baseline['time']:.2f}s")
        print(f"   {optimized['name']}: {optimized['time']:.2f}s")
        print(f"   🚀 Speedup: {speedup:.1f}x faster!")
        
        if speedup > 2:
            print("✅ Excellent speedup achieved!")
        elif speedup > 1.5:
            print("✅ Good speedup achieved!")
        else:
            print("⚠️  Limited speedup - check GPU/dependencies")
    
    # Print all transcriptions for comparison
    print(f"\n📝 Transcription Results:")
    for result in results:
        print(f"   {result['name']}: {result['transcription']}")

def test_real_time_capability():
    """Test real-time processing capability"""
    print("\n🎯 Real-time Capability Test")
    print("=" * 30)
    
    # Create optimized config for real-time
    real_time_config = ProcessingConfig(
        use_faster_whisper=True,
        faster_whisper_model="medium",  # Smaller model for speed
        compute_type="int8",
        beam_size=1,
        use_vad=True,
        vad_threshold=0.5,
        use_gpu=True
    )
    
    audio_config = AudioConfig(
        chunk_length_ms=10000,  # 10 second chunks for real-time feel
        overlap_ms=500,
        sample_rate=16000
    )
    
    print("⚙️  Real-time optimized settings:")
    print(f"   • Model: {real_time_config.faster_whisper_model}")
    print(f"   • Compute: {real_time_config.compute_type}")
    print(f"   • Beam size: {real_time_config.beam_size}")
    print(f"   • VAD enabled: {real_time_config.use_vad}")
    print(f"   • Chunk size: {audio_config.chunk_length_ms}ms")
    
    target_time = 2.0  # Target: process 10s audio in 2s for 5x real-time
    print(f"\n🎯 Target: Process audio 5x faster than real-time (<{target_time}s for 10s audio)")

if __name__ == "__main__":
    test_optimized_asr()
    test_real_time_capability()
    
    print("\n💡 Next Steps for Real-time System:")
    print("1. Use streaming/chunked processing")
    print("2. Implement audio buffer queue")
    print("3. Add voice activity detection")
    print("4. Pipeline ASR → LLM → TTS")
    print("5. Consider smaller models (medium/small) for even faster processing")
