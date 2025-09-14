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
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form
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

# Multi-agent LLM orchestration
multi_agent_import_error: Optional[str] = None
try:
    from MultiAgent import MultiAgent
except Exception as e:
    MultiAgent = None  # Will validate on first use
    multi_agent_import_error = str(e)

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
    model_id: str = "biodatlab-medium-faster"  # Default to recommended model
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
    llm_ready: Optional[bool] = None
    llm_model: Optional[str] = None
    llm_engine: Optional[str] = None


class LLMRequest(BaseModel):
    """Request model for /llm endpoint"""
    text: str
    history: Optional[List[Dict[str, Any]]] = None  # [{role, content}]


class LLMResponse(BaseModel):
    """Response model for /llm endpoint"""
    response: str
    model: Optional[str] = None
    used_base_url: Optional[str] = None
    timestamp: str
    status: str = "success"


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
multi_agent: Optional[Any] = None


# Initialize ASR model
def initialize_asr_model(model_id: str = "biodatlab-medium-faster"):
    """Initialize the Thai ASR model with GPU optimization"""
    global model_manager
    
    try:
        logger.info("🚀 Initializing Model Manager with GPU optimization...")
        logger.info("🔧 GPU Configuration:")
        logger.info("   - GPU Memory Fraction: 80%")
        logger.info("   - Compute Type: float16 (GPU optimized)")
        logger.info("   - Batch Size: 8 (increased for GPU efficiency)")
        logger.info("   - Workers: 4 (parallel processing)")
        logger.info("   - Chunk Length: 20s (optimized for GPU)")
        
        if model_manager is None:
            model_manager = get_model_manager()
        
        logger.info(f"📦 Loading model: requested_id='{model_id}'")
        model_manager.load_model(model_id)
        info = model_manager.get_current_model_info() or {}
        logger.info(
            f"📌 Loaded model resolved_id='{info.get('id','unknown')}', name='{info.get('name','')}', path='{info.get('model_path','')}'"
        )
        
        # Log GPU status
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
                logger.info(f"🎮 GPU Status: {gpu_count} GPU(s) available")
                logger.info(f"🎮 Primary GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                logger.info(f"💾 GPU Memory Utilization Target: 80%")
            else:
                logger.info("💻 No GPU available, using CPU optimization")
        except ImportError:
            logger.info("💻 PyTorch not available for GPU detection")
        
        logger.info("✅ Thai ASR model initialized successfully with GPU optimization!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ASR model: {e}")
        model_manager = None


# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🎬 Starting Multi-agent Call Center ..")
    logger.info(f"🧪 Python executable: {sys.executable}")
    logger.info(f"🧪 OPENROUTER_API_KEY set: {bool(os.getenv('OPENROUTER_API_KEY'))}")
    initialize_asr_model()
    
    # Create directories if they don't exist
    os.makedirs("audio_uploads", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    logger.info("🎉 Backend server started successfully!")

    # Lazy init MultiAgent so server can start even if CrewAI isn't installed
    global multi_agent
    if MultiAgent is not None:
        try:
            multi_agent = MultiAgent()
            logger.info("🧠 MultiAgent orchestrator initialized")
        except Exception as e:
            logger.warning(f"⚠️ MultiAgent init skipped: {e}")
    else:
        if multi_agent_import_error:
            logger.warning(f"⚠️ MultiAgent import failed: {multi_agent_import_error}")


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
    
    # LLM health (auto-init to ensure readiness reflected on health)
    llm_ready = False
    llm_model = None
    llm_engine = None
    global multi_agent
    if multi_agent is None and MultiAgent is not None:
        try:
            multi_agent = MultiAgent()
        except Exception as e:
            logger.warning(f"⚠️ LLM init during /health failed: {e}")
    if multi_agent is not None:
        try:
            st = multi_agent.get_status()
            llm_ready = bool(st.get("ready", False))
            llm_model = st.get("model")
            llm_engine = st.get("engine")
        except Exception:
            llm_ready = False

    return HealthResponse(
        status="healthy" if model_manager and model_manager.current_model else "degraded",
        timestamp=datetime.now().isoformat(),
        uptime=uptime,
        asr_model_loaded=model_manager is not None and model_manager.current_model is not None,
        device=current_model.get("device", "unknown") if current_model else "unknown",
        llm_ready=llm_ready,
        llm_model=llm_model,
        llm_engine=llm_engine,
    )


@app.get("/llm/health")
async def llm_health():
    """Dedicated LLM health endpoint; ensures LLM is initialized and reports status."""
    global multi_agent
    if multi_agent is None:
        if MultiAgent is None:
            raise HTTPException(
                status_code=503,
                detail=f"MultiAgent unavailable. Install crewai and litellm. Cause: {multi_agent_import_error}",
            )
        try:
            multi_agent = MultiAgent()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed initializing MultiAgent: {e}")

    try:
        st = multi_agent.get_status()
        return {
            "status": "ready" if st.get("ready") else "not_ready",
            "engine": st.get("engine"),
            "model": st.get("model"),
            "base_url": st.get("base_url"),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve LLM status: {e}")


# LLM multi-agent endpoint
@app.post("/llm", response_model=LLMResponse)
async def llm_generate(req: LLMRequest):
    """Generate a multi-agent LLM response for a given text input."""
    global multi_agent
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="'text' is required")

    # Ensure orchestrator exists (lazy init)
    if multi_agent is None:
        if MultiAgent is None:
            raise HTTPException(
                status_code=503,
                detail=f"MultiAgent unavailable. Install crewai and litellm, then restart server. Cause: {multi_agent_import_error}",
            )
        try:
            multi_agent = MultiAgent()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed initializing MultiAgent: {e}")

    try:
        logger.info("/llm called; generating response via MultiAgent")
        result = multi_agent.run(req.text, conversation_history=req.history)
        return LLMResponse(
            response=result.get("response", ""),
            model=result.get("model"),
            used_base_url=result.get("used_base_url"),
            timestamp=datetime.now().isoformat(),
            status="success",
        )
    except Exception as e:
        logger.error(f"❌ LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")


# Main ASR endpoint
@app.post("/api/asr", response_model=ASRResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form("th"),
    model_id: str = Form("biodatlab-medium-faster"),
    use_vad: bool = Form(True),
    beam_size: int = Form(1),
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
    logger.info(f"/api/asr called with model_id='{model_id}', language='{language}', file='{file.filename}'")
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
    try:
        resolved_id = model_manager.resolve_model_id(model_id)
    except Exception:
        resolved_id = None
    if not resolved_id:
        valid = ", ".join([m["id"] for m in model_manager.get_available_models()])
        raise HTTPException(status_code=400, detail=f"Unknown model_id '{model_id}'. Valid options: {valid}")

    if current_model_info and current_model_info.get("id") != resolved_id:
        logger.info(f"🔄 Switching model from {current_model_info.get('id')} to {model_id} (resolved='{resolved_id}')")
        try:
            model_manager.load_model(resolved_id)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load model '{model_id}' (resolved '{resolved_id}'): {str(e)}"
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
        used = model_manager.get_current_model_info() or {}
        logger.info(
            f"🧾 Transcribed with model_id='{used.get('id','unknown')}', name='{used.get('name','')}', path='{used.get('model_path','')}', device='{result.get('device','')}'"
        )
        
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
        
    except (KeyboardInterrupt, SystemExit) as e:
        logger.error(f"🛑 Server interrupt during transcription: {e}")
        raise HTTPException(status_code=500, detail="Server interrupted during transcription")
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
        logger.info(f"/api/models/{model_id}/load called")
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
    language: str = Form("th"),
    model_id: str = Form("biodatlab-medium-faster"),
    use_vad: bool = Form(True),
    beam_size: int = Form(1),
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
    try:
        resolved_id = model_manager.resolve_model_id(model_id)
    except Exception:
        resolved_id = None
    if not resolved_id:
        valid = ", ".join([m["id"] for m in model_manager.get_available_models()])
        raise HTTPException(status_code=400, detail=f"Unknown model_id '{model_id}'. Valid options: {valid}")
    if current_model_info and current_model_info.get("id") != resolved_id:
        try:
            model_manager.load_model(resolved_id)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load model '{model_id}' (resolved '{resolved_id}'): {str(e)}"
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
async def reload_asr_model(model_id: str = "biodatlab-medium-faster"):
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
            "reload": "/api/asr/reload",
            "llm": "/llm",
            "llm_health": "/llm/health"
        },
        "supported_models": [
            "biodatlab-faster (recommended)",
            "biodatlab-medium-faster (recommended)", 
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
