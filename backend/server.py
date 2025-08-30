#!/usr/bin/env python3
"""
Multi-agent Call Center Backend Server
Serves Thai ASR (Automatic Speech Recognition) API endpoint
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
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

# Import our model manager and ASR components
try:
    from whisper.model_manager import get_model_manager, ModelManager
    from whisper.faster_whisper_thai import FasterWhisperThai, WhisperConfig
except ImportError:
    # Fallback if module structure is different
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'whisper'))
    from whisper.model_manager import get_model_manager, ModelManager
    from faster_whisper_thai import FasterWhisperThai, WhisperConfig

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
    model_id: str = "biodatlab-faster"  # Default to recommended model
    use_vad: bool = True
    beam_size: int = 1


class ModelInfo(BaseModel):
    """Model information response"""
    id: str
    name: str
    type: str
    language: str
    description: str
    performance_tier: str
    recommended: bool


class ModelListResponse(BaseModel):
    """Response model for available models"""
    models: List[ModelInfo]
    current_model: Optional[Dict[str, Any]] = None


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
model_manager: Optional[ModelManager] = None
start_time = time.time()


# Initialize ASR model
def initialize_asr_model(model_id: str = "biodatlab-faster"):
    """Initialize the Thai ASR model"""
    global model_manager
    
    try:
        logger.info("🚀 Initializing Model Manager...")
        
        if model_manager is None:
            model_manager = get_model_manager()
        
        logger.info(f"📦 Loading model: {model_id}")
        model_manager.load_model(model_id)
        
        logger.info("✅ Thai ASR model initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ASR model: {e}")
        model_manager = None


# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🎬 Starting Multi-agent Call Center ..")
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
    
    current_model = model_manager.get_current_model_info() if model_manager else None
    
    return HealthResponse(
        status="healthy" if model_manager and model_manager.current_model else "degraded",
        timestamp=datetime.now().isoformat(),
        uptime=uptime,
        asr_model_loaded=model_manager is not None and model_manager.current_model is not None,
        device=current_model.get("device", "unknown") if current_model else "unknown"
    )


# Main ASR endpoint
@app.post("/api/asr", response_model=ASRResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = "th",
    model_id: str = "biodatlab-faster",
    use_vad: bool = True,
    beam_size: int = 1
):
    """
    Transcribe audio file to Thai text using selected model
    
    Args:
        file: Audio file (WAV, MP3, M4A, etc.)
        language: Language code (default: 'th' for Thai)
        model_id: Model ID to use for transcription
        use_vad: Use Voice Activity Detection (default: True)
        beam_size: Beam size for decoding (1-5, lower is faster)
        
    Returns:
        ASRResponse with transcription and metadata
    """
    if model_manager is None or model_manager.current_model is None:
        # Try to initialize with requested model
        try:
            initialize_asr_model(model_id)
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="ASR model not available. Please check server health."
            )
    
    # Check if we need to switch models
    current_model_info = model_manager.get_current_model_info()
    if current_model_info and current_model_info.get("id") != model_id:
        logger.info(f"🔄 Switching model from {current_model_info.get('id')} to {model_id}")
        try:
            model_manager.load_model(model_id)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load model '{model_id}': {str(e)}"
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
        
        # Transcribe audio using model manager
        logger.info("🎵 Starting transcription...")
        result = model_manager.transcribe_with_current_model(temp_file)
        
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


# Model-related endpoints

@app.get("/api/models", response_model=ModelListResponse)
async def get_available_models():
    """Get list of available ASR models"""
    if model_manager is None:
        model_manager_instance = get_model_manager()
    else:
        model_manager_instance = model_manager
    
    available_models = model_manager_instance.get_available_models()
    current_model = model_manager_instance.get_current_model_info()
    
    return ModelListResponse(
        models=[ModelInfo(**model) for model in available_models],
        current_model=current_model
    )


@app.post("/api/models/{model_id}/load")
async def load_model(model_id: str):
    """Load a specific ASR model"""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    try:
        model_manager.load_model(model_id)
        return {
            "status": "success", 
            "message": f"Model '{model_id}' loaded successfully",
            "model_info": model_manager.get_current_model_info()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


# Batch transcription endpoint
@app.post("/api/asr/batch")
async def transcribe_batch(
    files: list[UploadFile] = File(..., description="Audio files to transcribe"),
    language: str = "th",
    model_id: str = "biodatlab-faster",
    use_vad: bool = True,
    beam_size: int = 1
):
    """
    Transcribe multiple audio files
    
    Args:
        files: List of audio files
        language: Language code (default: 'th')
        model_id: Model ID to use for transcription
        use_vad: Use Voice Activity Detection
        beam_size: Beam size for decoding
        
    Returns:
        List of transcription results
    """
    if model_manager is None or model_manager.current_model is None:
        try:
            initialize_asr_model(model_id)
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="ASR model not available. Please check server health."
            )
    
    # Check if we need to switch models
    current_model_info = model_manager.get_current_model_info()
    if current_model_info and current_model_info.get("id") != model_id:
        try:
            model_manager.load_model(model_id)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load model '{model_id}': {str(e)}"
            )
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        try:
            # Transcribe each file individually
            audio_data = await file.read()
            
            with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
                tmp.write(audio_data)
                temp_file = tmp.name
            
            try:
                result = model_manager.transcribe_with_current_model(temp_file)
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
    if model_manager is None or model_manager.current_model is None:
        raise HTTPException(status_code=503, detail="ASR model not available")
    
    current_model_info = model_manager.get_current_model_info()
    if not current_model_info:
        raise HTTPException(status_code=503, detail="No model information available")
    
    return current_model_info


# Reload model endpoint (for development/debugging)
@app.post("/api/asr/reload")
async def reload_asr_model(model_id: str = "biodatlab-faster"):
    """Reload the ASR model (admin endpoint)"""
    try:
        logger.info(f"🔄 Reloading ASR model: {model_id}")
        initialize_asr_model(model_id)
        
        if model_manager is not None and model_manager.current_model is not None:
            return {
                "status": "success", 
                "message": f"ASR model '{model_id}' reloaded successfully",
                "model_info": model_manager.get_current_model_info()
            }
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
        "endpoints": {
            "asr": "/api/asr",
            "asr_batch": "/api/asr/batch",
            "asr_info": "/api/asr/info",
            "models": "/api/models",
            "load_model": "/api/models/{model_id}/load",
            "reload": "/api/asr/reload"
        },
        "supported_models": [
            "biodatlab-faster (recommended)",
            "pathumma-large (recommended)", 
            "large-v3-faster",
            "large-v3-standard",
            "medium-faster",
            "medium-standard"
        ]
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
