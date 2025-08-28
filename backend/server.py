#!/usr/bin/env python3
"""
Multi-agent Call Center Backend Server
Serves Thai ASR (Automatic Speech Recognition) API endpoint
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import time
import asyncio
from datetime import datetime

# Add backend to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# FastAPI and related imports
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our faster-whisper Thai ASR
from whisper.faster_whisper_thai import FasterWhisperThai, WhisperConfig, create_thai_asr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ASRResponse(BaseModel):
    """Response model for ASR API"""
    text: str
    language: str
    duration: float
    processing_time: float
    speed_ratio: float
    chunks_processed: int
    model: str
    device: str
    timestamp: str
    status: str = "success"


class ASRRequest(BaseModel):
    """Request model for ASR configuration"""
    language: str = "th"
    model_name: str = "large-v3"
    use_vad: bool = True
    beam_size: int = 1


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    uptime: float
    asr_model_loaded: bool
    device: str
    version: str = "1.0.0"


# Initialize FastAPI app
app = FastAPI(
    title="Multi-agent Call Center Backend",
    description="Thai ASR API for call center multi-agent system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
asr_model: Optional[FasterWhisperThai] = None
start_time = time.time()


# Initialize ASR model
def initialize_asr_model():
    """Initialize the Thai ASR model"""
    global asr_model
    
    try:
        logger.info("🚀 Initializing Thai ASR model...")
        
        # Create configuration for optimal performance
        config = WhisperConfig(
            model_name="large-v3",  # Best for Thai
            language="th",
            task="transcribe",
            device="auto",  # Auto-detect CUDA/CPU
            compute_type="int8_float16",  # Balanced speed/quality
            beam_size=1,  # Fast inference
            use_vad=True,  # Voice activity detection
            vad_threshold=0.35,
            chunk_length_ms=30000,  # 30 seconds optimal
            overlap_ms=1000
        )
        
        asr_model = create_thai_asr(config)
        logger.info("✅ Thai ASR model initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ASR model: {e}")
        asr_model = None


# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🎬 Starting Multi-agent Call Center Backend...")
    initialize_asr_model()
    
    # Create directories if they don't exist
    os.makedirs("audio_uploads", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    logger.info("🎉 Backend server started successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down backend server...")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if asr_model is not None else "degraded",
        timestamp=datetime.now().isoformat(),
        uptime=uptime,
        asr_model_loaded=asr_model is not None,
        device=asr_model.device if asr_model else "unknown"
    )


# Main ASR endpoint
@app.post("/api/asr", response_model=ASRResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = "th",
    use_vad: bool = True,
    beam_size: int = 1
):
    """
    Transcribe audio file to Thai text using faster-whisper
    
    Args:
        file: Audio file (WAV, MP3, M4A, etc.)
        language: Language code (default: 'th' for Thai)
        use_vad: Use Voice Activity Detection (default: True)
        beam_size: Beam size for decoding (1-5, lower is faster)
        
    Returns:
        ASRResponse with transcription and metadata
    """
    if asr_model is None:
        raise HTTPException(
            status_code=503,
            detail="ASR model not available. Please check server health."
        )
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file size (limit to 100MB)
    if file.size and file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size: 100MB")
    
    # Validate audio file extension
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    temp_file = None
    try:
        # Read audio data
        audio_data = await file.read()
        logger.info(f"📁 Received audio file: {file.filename} ({len(audio_data)} bytes)")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(audio_data)
            temp_file = tmp.name
        
        # Update ASR configuration if needed
        if (use_vad != asr_model.config.use_vad or 
            beam_size != asr_model.config.beam_size or
            language != asr_model.config.language):
            
            logger.info(f"🔧 Updating ASR config: lang={language}, vad={use_vad}, beam={beam_size}")
            asr_model.config.language = language
            asr_model.config.use_vad = use_vad
            asr_model.config.beam_size = beam_size
        
        # Transcribe audio
        logger.info("🎵 Starting transcription...")
        result = asr_model.transcribe(temp_file)
        
        # Create response
        response = ASRResponse(
            text=result["text"],
            language=result["language"],
            duration=result["duration"],
            processing_time=result["processing_time"],
            speed_ratio=result["speed_ratio"],
            chunks_processed=result["chunks_processed"],
            model=result["model"],
            device=result["device"],
            timestamp=datetime.now().isoformat(),
            status="success"
        )
        
        logger.info(f"✅ Transcription completed: {len(result['text'])} characters")
        return response
        
    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass


# Batch transcription endpoint
@app.post("/api/asr/batch")
async def transcribe_batch(
    files: list[UploadFile] = File(..., description="Audio files to transcribe"),
    language: str = "th",
    use_vad: bool = True,
    beam_size: int = 1
):
    """
    Transcribe multiple audio files
    
    Args:
        files: List of audio files
        language: Language code (default: 'th')
        use_vad: Use Voice Activity Detection
        beam_size: Beam size for decoding
        
    Returns:
        List of transcription results
    """
    if asr_model is None:
        raise HTTPException(
            status_code=503,
            detail="ASR model not available. Please check server health."
        )
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        try:
            # Transcribe each file individually
            # For now, we'll process sequentially. Could be optimized for parallel processing
            audio_data = await file.read()
            
            with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
                tmp.write(audio_data)
                temp_file = tmp.name
            
            try:
                result = asr_model.transcribe(temp_file)
                results.append({
                    "filename": file.filename,
                    "result": result,
                    "status": "success"
                })
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "error"
            })
    
    return {"results": results}


# Model information endpoint
@app.get("/api/asr/info")
async def get_asr_info():
    """Get information about the loaded ASR model"""
    if asr_model is None:
        raise HTTPException(status_code=503, detail="ASR model not available")
    
    return {
        "model_name": asr_model.config.model_name,
        "language": asr_model.config.language,
        "device": asr_model.device,
        "compute_type": asr_model.config.compute_type,
        "use_vad": asr_model.config.use_vad,
        "beam_size": asr_model.config.beam_size,
        "chunk_length_ms": asr_model.config.chunk_length_ms,
        "overlap_ms": asr_model.config.overlap_ms
    }


# Reload model endpoint (for development/debugging)
@app.post("/api/asr/reload")
async def reload_asr_model():
    """Reload the ASR model (admin endpoint)"""
    try:
        logger.info("🔄 Reloading ASR model...")
        initialize_asr_model()
        
        if asr_model is not None:
            return {"status": "success", "message": "ASR model reloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload ASR model")
    
    except Exception as e:
        logger.error(f"❌ Failed to reload ASR model: {e}")
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-agent Call Center Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "asr_endpoint": "/api/asr"
    }


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    # Development server
    logger.info("🚀 Starting development server...")
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
