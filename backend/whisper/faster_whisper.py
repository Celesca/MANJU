import os
from pydub import AudioSegment
import torch
import csv
import time
from tqdm import tqdm
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import numpy as np

# Force PyTorch backend to avoid TensorFlow issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

@dataclass
class AudioConfig:
    """Configuration for audio processing"""
    chunk_length_ms: int = 30000  # Increased for faster-whisper (30 seconds is optimal)
    overlap_ms: int = 1000  # Reduced overlap for speed
    min_chunk_length_ms: int = 1000
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class ProcessingConfig:
    """Configuration for processing pipeline"""
    model_name: str = "large-v3"  # Use standard model name for faster-whisper
    custom_model_path: Optional[str] = None  # Path to converted Thai model
    language: str = "th"
    task: str = "transcribe"
    
    # faster-whisper specific configs
    device: str = "cuda"  # "cuda", "cpu", or "auto"
    compute_type: str = "int8_float16"  # "int8", "int8_float16", "float16", "float32"
    beam_size: int = 1  # Reduced beam size for speed (1-5, lower is faster)
    best_of: int = 1  # Reduced for speed
    patience: float = 1.0
    temperature: float = 0.0
    
    # Batch processing (faster-whisper doesn't support batching like transformers)
    max_workers: int = 1  # Use 1 for GPU, more for CPU
    use_vad: bool = True  # Voice Activity Detection for speed
    vad_threshold: float = 0.35
    
    # Performance optimizations
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4


class AudioProcessor:
    """Handles audio file preprocessing and chunking"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
    
    def preprocess_audio(self, audio_path: str) -> List[str]:
        """
        Preprocess audio file into overlapping chunks optimized for faster-whisper
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of temporary file paths for audio chunks
        """
        try:
            # Check if ffmpeg is available
            try:
                # Load and optimize audio
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_channels(self.config.channels).set_frame_rate(self.config.sample_rate)
            except Exception as e:
                if "ffmpeg" in str(e).lower():
                    print(f"FFmpeg error: {e}")
                    print("💡 FFmpeg is required for audio processing.")
                    print("📥 Install FFmpeg:")
                    print("   1. Run: install_ffmpeg.bat")
                    print("   2. Or download from: https://github.com/BtbN/FFmpeg-Builds/releases")
                    print("   3. Add to PATH or set FFMPEG_BINARY environment variable")
                    return []
                else:
                    raise e
            
            # For faster-whisper, we can use larger chunks for better performance
            step_size = self.config.chunk_length_ms - self.config.overlap_ms
            chunk_files = []
            
            for start in range(0, len(audio), step_size):
                end = start + self.config.chunk_length_ms
                chunk = audio[start:end]
                
                # Skip chunks that are too short
                if len(chunk) < self.config.min_chunk_length_ms:
                    break
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    chunk.export(tmp_file.name, format="wav")
                    chunk_files.append(tmp_file.name)
            
            return chunk_files
            
        except Exception as e:
            print(f"Error preprocessing {audio_path}: {e}")
            return []
    
    def cleanup_temp_files(self, file_paths: List[str]) -> None:
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except Exception:
                pass


class FasterWhisperASR:
    """Faster-Whisper ASR pipeline wrapper - much faster than standard Whisper"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._setup_device()
        self._load_model()
    
    def _setup_device(self) -> None:
        """Setup device and compute type"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device
        
        # Adjust compute type based on device
        if self.device == "cpu":
            # CPU optimizations
            if self.config.compute_type in ["int8_float16", "float16"]:
                self.compute_type = "int8"  # CPU doesn't support float16
            else:
                self.compute_type = self.config.compute_type
        else:
            # GPU optimizations
            self.compute_type = self.config.compute_type
        
        print(f"🚀 Using device: {self.device} with compute type: {self.compute_type}")
    
    def _load_model(self) -> None:
        """Load the faster-whisper model"""
        try:
            from backend.whisper.faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not found. Install with: pip install faster-whisper\n"
                "This will provide 2-4x speed improvement over standard Whisper!"
            )
        
        # Check if we have a custom Thai model path
        if self.config.custom_model_path and os.path.exists(self.config.custom_model_path):
            model_name_or_path = self.config.custom_model_path
            print(f"📦 Loading custom Thai model from: {model_name_or_path}")
        else:
            model_name_or_path = self.config.model_name
            print(f"📦 Loading standard model: {model_name_or_path}")
            print("💡 To use the Thai NECTEC model with faster-whisper:")
            print("   1. Convert it to CTranslate2 format using ct2-transformers-converter")
            print("   2. Set custom_model_path in ProcessingConfig")
        
        # Load model with optimizations
        self.model = WhisperModel(
            model_name_or_path,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=4 if self.device == "cpu" else 0,
            num_workers=1,  # faster-whisper works best with 1 worker per model
        )
        
        print(f"✅ Model loaded successfully with {self.compute_type} precision")
    
    def transcribe_single(self, audio_file: str) -> str:
        """
        Transcribe a single audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcription text
        """
        try:
            # Configure transcription parameters for speed
            segments, info = self.model.transcribe(
                audio_file,
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
                no_speech_threshold=self.config.no_speech_threshold,
                logprob_threshold=self.config.log_prob_threshold,
                compression_ratio_threshold=self.config.compression_ratio_threshold,
            )
            
            # Combine all segments
            transcription = " ".join([segment.text.strip() for segment in segments])
            
            return transcription
            
        except Exception as e:
            print(f"Error transcribing {audio_file}: {e}")
            return f"[ERROR: {str(e)}]"
    
    def transcribe_chunks(self, chunk_files: List[str]) -> str:
        """
        Transcribe all chunks and combine results
        
        Args:
            chunk_files: List of audio chunk file paths
            
        Returns:
            Combined transcription
        """
        transcriptions = []
        
        # Process chunks sequentially (faster-whisper is already optimized)
        for chunk_file in chunk_files:
            transcription = self.transcribe_single(chunk_file)
            if transcription and not transcription.startswith("[ERROR"):
                transcriptions.append(transcription)
        
        return " ".join(transcriptions)


class AudioFileManager:
    """Manages audio file operations and results"""
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.results: List[Dict[str, str]] = []
    
    def is_directory(self) -> bool:
        """Check if input path is a directory"""
        return os.path.isdir(self.input_path)
    
    def is_file(self) -> bool:
        """Check if input path is a file"""
        return os.path.isfile(self.input_path)
    
    def get_audio_files(self, extension: str = ".wav") -> List[str]:
        """Get list of audio files"""
        if self.is_file():
            # Single file
            if self.input_path.lower().endswith(extension.lower()):
                return [os.path.basename(self.input_path)]
            else:
                # Allow any audio extension for single files
                return [os.path.basename(self.input_path)]
        elif self.is_directory():
            # Directory
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"Directory not found: {self.input_path}")
            
            return [f for f in sorted(os.listdir(self.input_path)) 
                    if f.lower().endswith(extension.lower())]
        else:
            raise FileNotFoundError(f"Path not found: {self.input_path}")
    
    def get_full_path(self, filename: str) -> str:
        """Get full path for a filename"""
        if self.is_file():
            return self.input_path
        else:
            return os.path.join(self.input_path, filename)
    
    def add_result(self, filename: str, transcription: str) -> None:
        """Add a transcription result"""
        file_id = os.path.splitext(filename)[0]
        self.results.append({"id": file_id, "transcription": transcription})
    
    def save_results(self, output_path: str) -> None:
        """Save results to CSV file"""
        with open(output_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["id", "transcription"])
            writer.writeheader()
            for row in tqdm(self.results, desc="Saving to CSV", unit="row"):
                writer.writerow(row)


class FastOverlappingASRPipeline:
    """Optimized pipeline using faster-whisper for 2-4x speed improvement"""
    
    def __init__(self, 
                 input_path: str,
                 audio_config: Optional[AudioConfig] = None,
                 processing_config: Optional[ProcessingConfig] = None):
        
        self.audio_config = audio_config or AudioConfig()
        self.processing_config = processing_config or ProcessingConfig()
        
        self.audio_processor = AudioProcessor(self.audio_config)
        self.asr = FasterWhisperASR(self.processing_config)
        self.file_manager = AudioFileManager(input_path)
        
        self.start_time = None
        self.end_time = None
        
        # Print optimization info
        self._print_optimization_info()
    
    def _print_optimization_info(self):
        """Print information about optimizations being used"""
        print("🚀 FastOverlappingASRPipeline Optimizations:")
        print(f"   • faster-whisper engine (2-4x speed improvement)")
        print(f"   • Device: {self.processing_config.device}")
        print(f"   • Compute type: {self.processing_config.compute_type}")
        print(f"   • Beam size: {self.processing_config.beam_size} (lower = faster)")
        print(f"   • VAD filtering: {'enabled' if self.processing_config.use_vad else 'disabled'}")
        print(f"   • Optimized chunk size: {self.audio_config.chunk_length_ms}ms")
        print(f"   • Reduced overlap: {self.audio_config.overlap_ms}ms")
    
    def __call__(self, audio_path: str = None) -> str:
        """
        Make the pipeline callable - transcribe single file
        
        Args:
            audio_path: Optional audio file path. If None, uses input_path from constructor
            
        Returns:
            Transcription text
        """
        if audio_path:
            # Create new file manager for the provided path
            temp_file_manager = AudioFileManager(audio_path)
            full_path = temp_file_manager.get_full_path(os.path.basename(audio_path))
        else:
            # Use existing file manager
            if not self.file_manager.is_file():
                raise ValueError("No single file specified. Provide audio_path or use a single file in constructor.")
            full_path = self.file_manager.input_path
        
        try:
            # For single file transcription, we can often process the whole file at once
            # if it's under 30 seconds, which is much faster
            try:
                # Try direct transcription first (fastest for shorter files)
                audio = AudioSegment.from_file(full_path)
                if len(audio) <= 30000:  # 30 seconds or less
                    return self.asr.transcribe_single(full_path)
            except:
                pass
            
            # Fall back to chunking for longer files
            chunk_files = self.audio_processor.preprocess_audio(full_path)
            
            if not chunk_files:
                return "[EMPTY FILE]"
            
            # Transcribe chunks
            transcription = self.asr.transcribe_chunks(chunk_files)
            
            # Cleanup temporary files
            self.audio_processor.cleanup_temp_files(chunk_files)
            
            return transcription
            
        except Exception as e:
            return f"[ERROR: {str(e)}]"
    
    def process_single_file(self, filename: str) -> Tuple[str, str]:
        """
        Process a single audio file with optimizations
        
        Args:
            filename: Name of the audio file
            
        Returns:
            Tuple of (filename, transcription)
        """
        full_path = self.file_manager.get_full_path(filename)
        
        try:
            # Optimization: Try direct transcription for shorter files
            try:
                audio = AudioSegment.from_file(full_path)
                if len(audio) <= 30000:  # 30 seconds or less - process directly
                    transcription = self.asr.transcribe_single(full_path)
                    return filename, transcription
            except:
                pass
            
            # For longer files, use chunking
            chunk_files = self.audio_processor.preprocess_audio(full_path)
            
            if not chunk_files:
                return filename, "[EMPTY FILE]"
            
            # Transcribe chunks
            transcription = self.asr.transcribe_chunks(chunk_files)
            
            # Cleanup temporary files
            self.audio_processor.cleanup_temp_files(chunk_files)
            
            return filename, transcription
            
        except Exception as e:
            return filename, f"[ERROR: {str(e)}]"
    
    def process_all_files(self, output_csv: str = "results.csv") -> Dict[str, Any]:
        """
        Process all audio files in the input folder
        
        Args:
            output_csv: Output CSV filename
            
        Returns:
            Processing statistics
        """
        self.start_time = time.perf_counter()
        
        # Get audio files
        wav_files = self.file_manager.get_audio_files()
        print(f"🎵 Found {len(wav_files)} audio files to process")
        
        if not wav_files:
            print("No audio files found!")
            return {}
        
        # For GPU processing with faster-whisper, sequential is often fastest
        # due to optimized memory usage and compute kernels
        print(f"🔄 Processing mode: Optimized sequential (faster-whisper)")
        
        # Process files sequentially (optimal for faster-whisper)
        self._process_sequential_optimized(wav_files)
        
        # Save results
        print(f"\n💾 Saving results to {output_csv}...")
        self.file_manager.save_results(output_csv)
        
        # Calculate statistics
        self.end_time = time.perf_counter()
        return self._get_statistics(output_csv)
    
    def _process_sequential_optimized(self, wav_files: List[str]) -> None:
        """Process files sequentially with faster-whisper optimizations"""
        for filename in tqdm(wav_files, desc="Processing files", unit="file"):
            start_file_time = time.perf_counter()
            
            filename_result, transcription = self.process_single_file(filename)
            self.file_manager.add_result(filename, transcription)
            
            end_file_time = time.perf_counter()
            file_time = end_file_time - start_file_time
            
            # Show progress with timing
            tqdm.write(f"✅ {filename} ({file_time:.2f}s): {transcription[:80]}{'...' if len(transcription) > 80 else ''}")
    
    def _get_statistics(self, output_csv: str) -> Dict[str, Any]:
        """Get processing statistics"""
        elapsed_time = self.end_time - self.start_time
        num_files = len(self.file_manager.results)
        
        stats = {
            "elapsed_time": elapsed_time,
            "output_file": output_csv,
            "total_files": num_files,
            "avg_time_per_file": elapsed_time / num_files if num_files > 0 else 0,
            "optimizations": [
                "faster-whisper engine (2-4x speed boost)",
                f"Optimized compute type ({self.processing_config.compute_type})",
                f"VAD filtering ({'enabled' if self.processing_config.use_vad else 'disabled'})",
                f"Reduced beam size ({self.processing_config.beam_size})",
                "Direct processing for short files",
                "Optimized chunking strategy",
                f"Device: {self.processing_config.device}"
            ]
        }
        
        # Print statistics
        print(f"\n✅ All done! Time taken: {elapsed_time:.2f} seconds")
        print(f"📄 Results saved to {output_csv}")
        print(f"📊 Total files processed: {num_files}")
        print(f"⏱️  Average time per file: {stats['avg_time_per_file']:.2f} seconds")
        print(f"🚀 Optimizations used:")
        for opt in stats['optimizations']:
            print(f"   • {opt}")
        
        return stats


# Compatibility wrapper - maintains the same interface as the original
class OverlappingASRPipeline(FastOverlappingASRPipeline):
    """
    Compatibility wrapper that maintains the same interface as the original
    but uses faster-whisper under the hood for improved performance
    """
    pass


# Helper function to convert NECTEC model to faster-whisper format
def convert_nectec_model_to_faster_whisper():
    """
    Instructions for converting the NECTEC Thai model to faster-whisper format
    """
    print("🔧 To convert the NECTEC Thai model for faster-whisper:")
    print("1. Install converter: pip install ct2-transformers-converter")
    print("2. Convert model:")
    print("   ct2-transformers-converter --model nectec/Pathumma-whisper-th-large-v3 \\")
    print("                              --output_dir ./models/nectec-thai-faster-whisper \\")
    print("                              --copy_files tokenizer.json preprocessor_config.json \\")
    print("                              --quantization int8_float16")
    print("3. Use the converted model:")
    print("   config = ProcessingConfig(custom_model_path='./models/nectec-thai-faster-whisper')")


# Example usage with optimized settings
if __name__ == "__main__":
    # Configuration optimized for speed
    input_folder = "path/to/your/audio/files"  # Change this to your audio folder
    
    # Speed-optimized audio config
    audio_config = AudioConfig(
        chunk_length_ms=30000,  # Larger chunks for faster-whisper
        overlap_ms=1000,  # Reduced overlap
        sample_rate=16000
    )
    
    # Speed-optimized processing config
    processing_config = ProcessingConfig(
        model_name="large-v3",  # Use large-v3 or convert NECTEC model
        custom_model_path=None,  # Set to converted NECTEC model path if available
        device="cuda",  # Use "cpu" if no GPU
        compute_type="int8_float16",  # Best balance of speed/quality for GPU
        beam_size=1,  # Fastest beam size
        use_vad=True,  # Enable VAD for speed
        language="th"
    )
    
    # Print conversion instructions for NECTEC model
    convert_nectec_model_to_faster_whisper()
    
    # Create and run pipeline
    pipeline = FastOverlappingASRPipeline(
        input_path=input_folder,
        audio_config=audio_config,
        processing_config=processing_config
    )
    
    # Process all files
    stats = pipeline.process_all_files("output_results.csv")
