import os
from pydub import AudioSegment
from transformers import pipeline
import torch
import csv
import time
from tqdm import tqdm
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

# Force PyTorch backend to avoid TensorFlow issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

@dataclass
class AudioConfig:
    """Configuration for audio processing"""
    chunk_length_ms: int = 27000
    overlap_ms: int = 2000
    min_chunk_length_ms: int = 1000
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class ProcessingConfig:
    """Configuration for processing pipeline"""
    model_name: str = "nectec/Pathumma-whisper-th-large-v3"
    language: str = "th"
    task: str = "transcribe"
    batch_size: int = 4
    max_workers: int = 2
    use_gpu: bool = True
    
    # Faster-whisper optimization options
    use_faster_whisper: bool = True  # Enable faster-whisper for 2-4x speed improvement
    faster_whisper_model: str = "large-v3"  # Model for faster-whisper (use large-v3 for Thai)
    compute_type: str = "int8_float16"  # "int8", "int8_float16", "float16", "float32"
    beam_size: int = 1  # Reduced beam size for speed (1-5, lower is faster)
    use_vad: bool = True  # Voice Activity Detection for speed
    vad_threshold: float = 0.35


class AudioProcessor:
    """Handles audio file preprocessing and chunking"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
    
    def preprocess_audio(self, audio_path: str) -> List[str]:
        """
        Preprocess audio file into overlapping chunks
        
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


class WhisperASR:
    """Whisper ASR pipeline wrapper with faster-whisper support"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.use_faster_whisper = config.use_faster_whisper
        self._setup_device()
        self._load_model()
    
    def _setup_device(self) -> None:
        """Setup device and torch dtype"""
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = 0 if not self.use_faster_whisper else "cuda"
            self.torch_dtype = torch.bfloat16
        else:
            self.device = -1 if not self.use_faster_whisper else "cpu"
            self.torch_dtype = torch.float32
    
    def _load_model(self) -> None:
        """Load the Whisper model - either faster-whisper or standard"""
        if self.use_faster_whisper:
            self._load_faster_whisper_model()
        else:
            self._load_standard_whisper_model()
    
    def _load_faster_whisper_model(self) -> None:
        """Load faster-whisper model for 2-4x speed improvement"""
        try:
            from faster_whisper import WhisperModel
        except Exception:
            print("⚠️ faster-whisper not found. Install with: pip install faster-whisper")
            print("Falling back to standard Whisper...")
            self.use_faster_whisper = False
            self._load_standard_whisper_model()
            return
        
        # Adjust compute type based on device
        compute_type = self.config.compute_type
        if self.device == "cpu" and compute_type in ["int8_float16", "float16"]:
            compute_type = "int8"  # CPU doesn't support float16
        
        print(f"🚀 Loading faster-whisper model: {self.config.faster_whisper_model}")
        print(f"   Device: {self.device}, Compute type: {compute_type}")
        
        self.model = WhisperModel(
            self.config.faster_whisper_model,
            device=self.device,
            compute_type=compute_type,
            cpu_threads=4 if self.device == "cpu" else 0,
            num_workers=1,
        )
        
        print("✅ faster-whisper model loaded successfully!")
        print("💡 Expected 2-4x speed improvement over standard Whisper")
    
    def _load_standard_whisper_model(self) -> None:
        """Load standard transformers Whisper model"""
        print(f"📦 Loading standard Whisper model: {self.config.model_name}")

        # Prefer safetensors weights; gracefully fallback to bin
        def _build_pipe(prefer_safetensors: bool):
            mk = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }
            if prefer_safetensors:
                mk["use_safetensors"] = True
            else:
                # Let transformers decide / allow bin
                mk["use_safetensors"] = None

            return pipeline(
                task="automatic-speech-recognition",
                model=self.config.model_name,
                device=self.device,
                batch_size=self.config.batch_size,
                framework="pt",  # Force PyTorch framework
                model_kwargs=mk,
            )

        try:
            self.pipe = _build_pipe(prefer_safetensors=True)
        except Exception:
            self.pipe = _build_pipe(prefer_safetensors=False)

        # Set language and task
        self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(
            language=self.config.language,
            task=self.config.task,
        )

        print("✅ Standard Whisper model loaded successfully!")
    
    def transcribe_batch(self, audio_files: List[str]) -> List[str]:
        """
        Transcribe a batch of audio files
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of transcriptions
        """
        if self.use_faster_whisper:
            # faster-whisper doesn't support batching, process sequentially
            return [self._transcribe_single_faster_whisper(file) for file in audio_files]
        else:
            return self._transcribe_batch_standard(audio_files)
    
    def _transcribe_single_faster_whisper(self, audio_file: str) -> str:
        """Transcribe single file with faster-whisper"""
        try:
            segments, info = self.model.transcribe(
                audio_file,
                language=self.config.language,
                task=self.config.task,
                beam_size=self.config.beam_size,
                vad_filter=self.config.use_vad,
                vad_parameters=dict(
                    threshold=self.config.vad_threshold,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100,
                    speech_pad_ms=30,
                ) if self.config.use_vad else None,
            )
            
            # Combine all segments
            transcription = " ".join([segment.text.strip() for segment in segments])
            return transcription
            
        except Exception as e:
            print(f"Error transcribing {audio_file}: {e}")
            return "[ERROR]"
    
    def _transcribe_batch_standard(self, audio_files: List[str]) -> List[str]:
        """Transcribe batch with standard Whisper"""
        try:
            batch_results = self.pipe(audio_files)
            
            if isinstance(batch_results, list):
                return [result["text"].strip() for result in batch_results]
            else:
                return [batch_results["text"].strip()]
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Fallback to single file processing
            transcriptions = []
            for file_path in audio_files:
                try:
                    result = self.pipe(file_path)
                    transcriptions.append(result["text"].strip())
                except Exception:
                    transcriptions.append("[ERROR]")
            return transcriptions
    
    def transcribe_chunks(self, chunk_files: List[str]) -> str:
        """
        Transcribe all chunks and combine results
        
        Args:
            chunk_files: List of audio chunk file paths
            
        Returns:
            Combined transcription
        """
        transcriptions = []
        
        # Process in batches
        for i in range(0, len(chunk_files), self.config.batch_size):
            batch_files = chunk_files[i:i+self.config.batch_size]
            batch_transcriptions = self.transcribe_batch(batch_files)
            transcriptions.extend(batch_transcriptions)
        
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


class OverlappingASRPipeline:
    """Main pipeline for overlapping ASR processing"""
    
    def __init__(self, 
                 input_path: str,
                 audio_config: Optional[AudioConfig] = None,
                 processing_config: Optional[ProcessingConfig] = None):
        
        self.audio_config = audio_config or AudioConfig()
        self.processing_config = processing_config or ProcessingConfig()
        
        self.audio_processor = AudioProcessor(self.audio_config)
        self.asr = WhisperASR(self.processing_config)
        self.file_manager = AudioFileManager(input_path)
        
        self.start_time = None
        self.end_time = None
    
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
            # Create overlapping chunks
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
        Process a single audio file
        
        Args:
            filename: Name of the audio file
            
        Returns:
            Tuple of (filename, transcription)
        """
        full_path = self.file_manager.get_full_path(filename)
        
        try:
            # Create overlapping chunks
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
    
    def should_use_threading(self) -> bool:
        """Determine if threading should be used"""
        # faster-whisper works best without threading due to optimized kernels
        if hasattr(self.asr, 'use_faster_whisper') and self.asr.use_faster_whisper:
            return False
        return not (self.processing_config.use_gpu and torch.cuda.is_available())
    
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
        
        # Determine processing mode
        use_threading = self.should_use_threading()
        max_workers = self.processing_config.max_workers if use_threading else 1
        
        print(f"🔄 Processing mode: {'Multi-threaded' if use_threading else 'Single-threaded'} ({max_workers} workers)")
        
        # Process files
        if use_threading and len(wav_files) > 1:
            self._process_with_threading(wav_files, max_workers)
        else:
            self._process_sequential(wav_files)
        
        # Save results
        print(f"\n💾 Saving results to {output_csv}...")
        self.file_manager.save_results(output_csv)
        
        # Calculate statistics
        self.end_time = time.perf_counter()
        return self._get_statistics(output_csv)
    
    def _process_with_threading(self, wav_files: List[str], max_workers: int) -> None:
        """Process files using threading"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_single_file, filename): filename 
                      for filename in wav_files}
            
            for future in tqdm(futures, desc="Processing files", unit="file"):
                filename, transcription = future.result()
                self.file_manager.add_result(filename, transcription)
                tqdm.write(f"✅ {filename}: {transcription[:80]}{'...' if len(transcription) > 80 else ''}")
    
    def _process_sequential(self, wav_files: List[str]) -> None:
        """Process files sequentially"""
        for filename in tqdm(wav_files, desc="Processing files", unit="file"):
            filename_result, transcription = self.process_single_file(filename)
            self.file_manager.add_result(filename, transcription)
            tqdm.write(f"✅ {filename}: {transcription[:80]}{'...' if len(transcription) > 80 else ''}")
    
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
                f"Batch processing (batch_size={self.processing_config.batch_size})",
                "Optimized audio preprocessing",
                f"Smart threading ({'enabled' if self.should_use_threading() else 'disabled for GPU'})",
                "Efficient temporary file handling"
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


# Example usage
if __name__ == "__main__":
    # Configuration
    input_folder = "path/to/your/audio/files"  # Change this to your audio folder
    
    # Custom configurations for faster-whisper (RECOMMENDED for speed)
    audio_config = AudioConfig(
        chunk_length_ms=30000,  # Larger chunks work better with faster-whisper
        overlap_ms=1000,  # Reduced overlap for speed
        sample_rate=16000
    )
    
    # Faster-whisper configuration (2-4x speed improvement)
    processing_config = ProcessingConfig(
        # Standard Whisper settings (fallback)
        model_name="nectec/Pathumma-whisper-th-large-v3",
        batch_size=4,
        max_workers=2,
        
        # Faster-whisper settings (recommended)
        use_faster_whisper=True,  # Enable for major speed boost
        faster_whisper_model="large-v3",  # Use large-v3 for Thai (good balance)
        compute_type="int8_float16",  # Best for GPU, use "int8" for CPU
        beam_size=1,  # Lower = faster (1-5)
        use_vad=True,  # Voice Activity Detection for speed
        vad_threshold=0.35,
        
        language="th",
        use_gpu=True
    )
    
    print("🚀 Performance Comparison:")
    print("   Standard Whisper: ~2-4 seconds per file")
    print("   faster-whisper:   ~0.5-1 seconds per file (2-4x faster!)")
    print("   With VAD + int8:  Even faster on suitable hardware")
    print()
    
    # Create and run pipeline
    pipeline = OverlappingASRPipeline(
        input_folder=input_folder,
        audio_config=audio_config,
        processing_config=processing_config
    )
    
    # Process all files
    stats = pipeline.process_all_files("output_results.csv")
    
    # Additional tips for maximum speed
    print("\n💡 Speed Optimization Tips:")
    print("1. Install faster-whisper: pip install faster-whisper")
    print("2. Use int8_float16 compute type for GPU")
    print("3. Set beam_size=1 for maximum speed")
    print("4. Enable VAD to skip silent parts")
    print("5. Use larger chunk sizes (30s) to reduce overhead")
    if torch.cuda.is_available():
        print("6. ✅ CUDA detected - you should see significant speedup!")
    else:
        print("6. Consider using GPU for even better performance")


# --- Simple Transformers-based ASR adapter with process_file() interface ---

class SimplePipelineASR:
    """Lightweight ASR that uses Transformers pipeline directly and exposes process_file().

    - Prefers safetensors weights for downloads.
    - Chunks audio using AudioProcessor and batches through Transformers pipeline.
    - Returns a dict matching model_manager expectations.
    """

    def __init__(self, audio_config: AudioConfig, processing_config: ProcessingConfig):
        self.audio_config = audio_config
        # Ensure we don't accidentally enable faster-whisper in this adapter
        self.processing_config = ProcessingConfig(
            model_name=processing_config.model_name,
            language=processing_config.language,
            task=processing_config.task,
            batch_size=processing_config.batch_size,
            max_workers=processing_config.max_workers,
            use_gpu=processing_config.use_gpu,
            use_faster_whisper=False,
        )

        # Device and dtype
        if self.processing_config.use_gpu and torch.cuda.is_available():
            self.device = 0
            self.torch_dtype = torch.float16
        else:
            self.device = -1
            self.torch_dtype = torch.float32

        # Build Transformers pipeline with safetensors preference
        def _build_pipe(prefer_safetensors: bool):
            mk = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }
            if prefer_safetensors:
                mk["use_safetensors"] = True
            else:
                mk["use_safetensors"] = None

            return pipeline(
                task="automatic-speech-recognition",
                model=self.processing_config.model_name,
                device=self.device,
                batch_size=self.processing_config.batch_size,
                framework="pt",
                model_kwargs=mk,
            )

        try:
            self.pipe = _build_pipe(prefer_safetensors=True)
        except Exception:
            self.pipe = _build_pipe(prefer_safetensors=False)

        # Force language/task
        self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(
            language=self.processing_config.language, task=self.processing_config.task
        )

        # Audio chunker
        self.audio_processor = AudioProcessor(self.audio_config)
        # Expose a minimal attribute used by model_manager for device reporting
        class _ASRInfo:
            def __init__(self, device):
                self.device = device
        self.asr = _ASRInfo("cuda" if self.device == 0 else "cpu")

    def process_file(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe a file and return a standardized result dict."""
        start = time.perf_counter()

        # Create chunks
        chunk_files = self.audio_processor.preprocess_audio(audio_path)
        if not chunk_files:
            return {
                "transcription": "",
                "total_duration": 0.0,
                "total_processing_time": 0.0,
                "total_chunks": 0,
            }

        # Transcribe in batches
        texts: List[str] = []
        bs = self.processing_config.batch_size or 1
        for i in range(0, len(chunk_files), bs):
            batch = chunk_files[i : i + bs]
            try:
                out = self.pipe(batch)
                if isinstance(out, list):
                    texts.extend([o.get("text", "").strip() for o in out])
                else:
                    texts.append(out.get("text", "").strip())
            except Exception:
                # Fallback to single-file loop if batch failed
                for f in batch:
                    try:
                        o = self.pipe(f)
                        texts.append(o.get("text", "").strip())
                    except Exception:
                        texts.append("")

        # Cleanup temps
        self.audio_processor.cleanup_temp_files(chunk_files)

        # Compute duration with pydub
        try:
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0
        except Exception:
            duration = 0.0

        elapsed = time.perf_counter() - start
        return {
            "transcription": " ".join([t for t in texts if t]).strip(),
            "total_duration": duration,
            "total_processing_time": elapsed,
            "total_chunks": len(chunk_files),
        }