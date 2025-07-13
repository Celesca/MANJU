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
            # Load and optimize audio
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(self.config.channels).set_frame_rate(self.config.sample_rate)
            
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
    """Whisper ASR pipeline wrapper"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._setup_device()
        self._load_model()
    
    def _setup_device(self) -> None:
        """Setup device and torch dtype"""
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = 0
            self.torch_dtype = torch.bfloat16
        else:
            self.device = -1
            self.torch_dtype = torch.float32
    
    def _load_model(self) -> None:
        """Load the Whisper model"""
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.config.model_name,
            torch_dtype=self.torch_dtype,
            device=self.device,
            batch_size=self.config.batch_size,
            framework="pt"  # Force PyTorch framework
        )
        
        # Set language and task
        self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(
            language=self.config.language, 
            task=self.config.task
        )
    
    def transcribe_batch(self, audio_files: List[str]) -> List[str]:
        """
        Transcribe a batch of audio files
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of transcriptions
        """
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
    
    # Custom configurations (optional)
    audio_config = AudioConfig(
        chunk_length_ms=27000,
        overlap_ms=2000,
        sample_rate=16000
    )
    
    processing_config = ProcessingConfig(
        model_name="nectec/Pathumma-whisper-th-large-v3",
        batch_size=4,
        max_workers=2
    )
    
    # Create and run pipeline
    pipeline = OverlappingASRPipeline(
        input_folder=input_folder,
        audio_config=audio_config,
        processing_config=processing_config
    )
    
    # Process all files
    stats = pipeline.process_all_files("output_results.csv")