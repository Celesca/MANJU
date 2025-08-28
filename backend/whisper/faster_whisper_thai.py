#!/usr/bin/env python3
"""
Optimized faster-whisper implementation for Thai language ASR
Provides 2-4x speed improvement over standard Whisper transformers
"""

import os
import tempfile
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from pydub import AudioSegment

# Force PyTorch backend to avoid TensorFlow issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

@dataclass
class WhisperConfig:
    """Configuration for faster-whisper Thai ASR"""
    
    # Model configuration
    model_name: str = "large-v3"  # Best model for Thai language
    language: str = "th"
    task: str = "transcribe"
    
    # Device and compute settings
    device: str = "auto"  # "cuda", "cpu", or "auto"
    compute_type: str = "int8_float16"  # "int8", "int8_float16", "float16", "float32"
    
    # Audio processing
    chunk_length_ms: int = 30000  # 30 seconds optimal for faster-whisper
    overlap_ms: int = 1000  # Minimal overlap for speed
    sample_rate: int = 16000
    channels: int = 1
    
    # Transcription parameters
    beam_size: int = 1  # Lower for speed, higher for accuracy
    best_of: int = 1
    patience: float = 1.0
    temperature: float = 0.0
    
    # VAD (Voice Activity Detection) settings
    use_vad: bool = True
    vad_threshold: float = 0.35
    
    # Quality thresholds
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4


class FasterWhisperThai:
    """Optimized faster-whisper for Thai language ASR"""
    
    def __init__(self, config: WhisperConfig = None):
        self.config = config or WhisperConfig()
        self.model = None
        self._setup_device()
        self._load_model()
    
    def _setup_device(self):
        """Setup computing device"""
        if self.config.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    print("🚀 Using CUDA GPU acceleration")
                else:
                    self.device = "cpu"
                    print("💻 Using CPU (consider GPU for better performance)")
            except ImportError:
                self.device = "cpu"
                print("💻 Using CPU")
        else:
            self.device = self.config.device
        
        # Adjust compute type for CPU
        if self.device == "cpu" and self.config.compute_type in ["int8_float16", "float16"]:
            self.config.compute_type = "int8"
            print("⚠️ Adjusted compute_type to 'int8' for CPU compatibility")
    
    def _load_model(self):
        """Load the faster-whisper model"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            )
        
        print(f"📦 Loading faster-whisper model: {self.config.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Compute type: {self.config.compute_type}")
        
        try:
            self.model = WhisperModel(
                self.config.model_name,
                device=self.device,
                compute_type=self.config.compute_type,
                cpu_threads=4 if self.device == "cpu" else 0,
                num_workers=1,
                download_root=None,  # Use default cache
                local_files_only=False
            )
            print("✅ faster-whisper model loaded successfully!")
            print("💡 Expected 2-4x speed improvement over standard Whisper")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio file to optimal format for faster-whisper
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to preprocessed audio file
        """
        try:
            # Load audio with pydub
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to optimal format
            audio = audio.set_channels(self.config.channels)
            audio = audio.set_frame_rate(self.config.sample_rate)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                audio.export(tmp_file.name, format="wav")
                return tmp_file.name
                
        except Exception as e:
            print(f"❌ Error preprocessing audio: {e}")
            raise
    
    def _chunk_audio(self, audio_path: str) -> List[str]:
        """
        Split audio into overlapping chunks for processing
        
        Args:
            audio_path: Path to preprocessed audio file
            
        Returns:
            List of chunk file paths
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            
            # If audio is short enough, return as single chunk
            if len(audio) <= self.config.chunk_length_ms:
                return [audio_path]
            
            chunk_files = []
            step_size = self.config.chunk_length_ms - self.config.overlap_ms
            
            for start in range(0, len(audio), step_size):
                end = start + self.config.chunk_length_ms
                chunk = audio[start:end]
                
                # Skip very short chunks
                if len(chunk) < 1000:  # 1 second minimum
                    break
                
                # Create temporary chunk file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    chunk.export(tmp_file.name, format="wav")
                    chunk_files.append(tmp_file.name)
            
            return chunk_files
            
        except Exception as e:
            print(f"❌ Error chunking audio: {e}")
            return [audio_path]  # Fallback to original file
    
    def _transcribe_chunk(self, audio_path: str) -> str:
        """
        Transcribe a single audio chunk
        
        Args:
            audio_path: Path to audio chunk
            
        Returns:
            Transcription text
        """
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=self.config.language,
                task=self.config.task,
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                patience=self.config.patience,
                temperature=self.config.temperature,
                vad_filter=self.config.use_vad,
                vad_parameters=dict(
                    threshold=self.config.vad_threshold,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100,
                    speech_pad_ms=30,
                ) if self.config.use_vad else None,
                without_timestamps=False,
                no_speech_threshold=self.config.no_speech_threshold,
                log_prob_threshold=self.config.log_prob_threshold,
                compression_ratio_threshold=self.config.compression_ratio_threshold,
            )
            
            # Combine all segments
            text_segments = []
            for segment in segments:
                if segment.text.strip():
                    text_segments.append(segment.text.strip())
            
            return " ".join(text_segments)
            
        except Exception as e:
            print(f"❌ Error transcribing chunk: {e}")
            return ""
    
    def _cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass  # Ignore cleanup errors
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file to Thai text
        
        Args:
            audio_path: Path to audio file (supports wav, mp3, m4a, etc.)
            
        Returns:
            Dictionary with transcription results and metadata
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        start_time = time.time()
        temp_files = []
        
        try:
            print(f"🎵 Transcribing: {os.path.basename(audio_path)}")
            
            # Preprocess audio
            preprocessed_path = self._preprocess_audio(audio_path)
            temp_files.append(preprocessed_path)
            
            # Split into chunks if needed
            chunk_paths = self._chunk_audio(preprocessed_path)
            temp_files.extend(chunk_paths)
            
            # Transcribe all chunks
            transcriptions = []
            for i, chunk_path in enumerate(chunk_paths):
                print(f"   Processing chunk {i+1}/{len(chunk_paths)}...")
                chunk_text = self._transcribe_chunk(chunk_path)
                if chunk_text:
                    transcriptions.append(chunk_text)
            
            # Combine all transcriptions
            full_text = " ".join(transcriptions)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get audio duration
            audio = AudioSegment.from_file(audio_path)
            audio_duration = len(audio) / 1000.0  # Convert to seconds
            
            result = {
                "text": full_text,
                "language": self.config.language,
                "duration": audio_duration,
                "processing_time": processing_time,
                "speed_ratio": audio_duration / processing_time if processing_time > 0 else 0,
                "chunks_processed": len(chunk_paths),
                "model": self.config.model_name,
                "device": self.device
            }
            
            print(f"✅ Transcription completed in {processing_time:.2f}s")
            print(f"   Speed ratio: {result['speed_ratio']:.1f}x realtime")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            raise
        
        finally:
            # Clean up temporary files
            self._cleanup_temp_files(temp_files)
    
    def transcribe_audio_data(self, audio_data: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        """
        Transcribe audio from bytes data (useful for API endpoints)
        
        Args:
            audio_data: Raw audio bytes
            filename: Original filename (for format detection)
            
        Returns:
            Dictionary with transcription results and metadata
        """
        temp_file = None
        try:
            # Save audio data to temporary file
            suffix = Path(filename).suffix.lower() or '.wav'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(audio_data)
                temp_file = tmp_file.name
            
            # Transcribe using the file-based method
            return self.transcribe(temp_file)
            
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass


def create_thai_asr(config: WhisperConfig = None) -> FasterWhisperThai:
    """
    Factory function to create Thai ASR instance
    
    Args:
        config: Optional configuration, uses defaults if None
        
    Returns:
        Configured FasterWhisperThai instance
    """
    return FasterWhisperThai(config)


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing faster-whisper Thai ASR")
    
    # Create ASR instance with default config
    asr = create_thai_asr()
    
    # Example transcription (replace with actual audio file)
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        try:
            result = asr.transcribe(test_file)
            print(f"\n📝 Transcription: {result['text']}")
            print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
            print(f"🚀 Speed ratio: {result['speed_ratio']:.1f}x realtime")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print(f"ℹ️ Test file '{test_file}' not found. Place an audio file to test.")
