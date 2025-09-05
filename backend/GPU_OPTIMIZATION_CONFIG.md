# GPU Optimization Configuration for Thai ASR FastAPI Backend

## Overview
This document describes the GPU optimization settings implemented to achieve 80% GPU utilization for improved FastAPI performance in the Thai ASR system.

## Key Optimizations Applied

### 1. Memory Management
- **GPU Memory Fraction**: Set to 0.8 (80% GPU memory utilization)
- **Memory Allocation**: Pre-allocated using `torch.cuda.set_per_process_memory_fraction(0.8)`
- **Compute Type**: Changed from `int8_float16` to `float16` for better GPU performance

### 2. Threading and Workers
- **CPU Threads**: Increased from 4 to 8 for better CPU utilization
- **Num Workers**: Increased from 1-2 to 4 for better parallel processing
- **Batch Size**: Increased from 4 to 8 for more efficient GPU batch processing

### 3. Audio Processing Optimization
- **Chunk Length**: Reduced from 30s to 20s for better parallelization
- **Overlap**: Reduced from 1000ms to 500ms for faster processing
- **VAD Threshold**: Reduced from 0.35 to 0.3 for more aggressive voice detection

### 4. CUDA Optimizations
- **Flash Attention**: Enabled `torch.backends.cuda.enable_flash_sdp(True)`
- **CuDNN Benchmark**: Enabled `torch.backends.cudnn.benchmark = True`
- **Deterministic**: Disabled `torch.backends.cudnn.deterministic = False` for speed

## Configuration Files Modified

### 1. `whisper/faster_whisper_thai.py`
```python
@dataclass
class WhisperConfig:
    compute_type: str = "float16"  # Optimized for GPU
    gpu_memory_fraction: float = 0.8  # 80% GPU utilization
    num_workers: int = 4  # Increased workers
    cpu_threads: int = 8  # Optimized threading
    chunk_length_ms: int = 20000  # Reduced chunk size
    overlap_ms: int = 500  # Reduced overlap
    batch_size: int = 8  # Increased batch size
```

### 2. `whisper/model_manager.py`
- Updated faster-whisper config with GPU optimization parameters
- Updated standard whisper config with increased batch size and workers

### 3. `whisper/whisper.py`
```python
@dataclass
class ProcessingConfig:
    batch_size: int = 8  # Increased for GPU
    max_workers: int = 4  # Increased workers
    gpu_memory_fraction: float = 0.8  # 80% GPU utilization
    compute_type: str = "float16"  # GPU optimized
```

## Expected Performance Improvements

### Speed Improvements
- **2-4x faster** processing with faster-whisper optimization
- **30-50% faster** audio preprocessing with optimized chunking
- **Better GPU utilization** leading to reduced processing time per request

### Memory Efficiency
- **80% GPU memory utilization** ensures maximum hardware usage
- **Optimized batch processing** reduces memory fragmentation
- **Better parallelization** across multiple workers

## Usage Notes

### GPU Requirements
- **NVIDIA GPU** with CUDA support required for optimal performance
- **Minimum 6GB VRAM** recommended for 80% utilization
- **CUDA 11.8+** or **CUDA 12.x** for best compatibility

### Monitoring GPU Usage
You can monitor GPU utilization using:
```bash
nvidia-smi
# or
watch -n 1 nvidia-smi
```

### Fallback Behavior
- If GPU is not available, the system automatically falls back to CPU with optimized settings
- CPU-specific optimizations include adjusted compute types and increased thread counts

## Environment Variables (Optional)
You can override default settings using environment variables:
```bash
# Set custom GPU memory fraction
export GPU_MEMORY_FRACTION=0.8

# Set custom batch size
export BATCH_SIZE=8

# Set custom number of workers
export NUM_WORKERS=4
```

## Troubleshooting

### Common Issues
1. **Out of Memory (OOM) Errors**
   - Reduce `gpu_memory_fraction` from 0.8 to 0.6 or 0.7
   - Reduce `batch_size` from 8 to 4 or 6

2. **Slow Performance**
   - Verify GPU is being used: Check server startup logs for "Using CUDA GPU acceleration"
   - Ensure CUDA drivers are properly installed
   - Monitor GPU utilization with `nvidia-smi`

3. **Model Loading Issues**
   - Check if sufficient GPU memory is available
   - Verify model files are properly downloaded
   - Check for CUDA compatibility issues

### Performance Tuning
- **High-end GPUs (16GB+ VRAM)**: Can increase batch_size to 12-16
- **Mid-range GPUs (8-12GB VRAM)**: Current settings (batch_size=8) are optimal
- **Low-end GPUs (4-6GB VRAM)**: Reduce batch_size to 4, gpu_memory_fraction to 0.6

## Verification

To verify the optimizations are working:

1. **Check server startup logs** for GPU optimization messages
2. **Monitor nvidia-smi** during transcription requests
3. **Test transcription speed** before and after optimization
4. **Check memory usage** to ensure 80% utilization is achieved

## Contact
For performance tuning questions or issues, refer to the model documentation or check the server logs for detailed error messages.
