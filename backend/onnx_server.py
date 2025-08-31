#!/usr/bin/env python3
"""
ONNX-based Multi-agent Call Center Backend Server
Serves Thai ASR using Sherpa-ONNX for faster inference
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

# ONNX and audio processing imports
try:
    import sherpa_onnx
    import numpy as np
    import soundfile as sf
    from pydub import AudioSegment
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    sherpa_onnx = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ONNXASRResponse(BaseModel):
    """Response model for ONNX ASR API"""
    text: str
    language: str
    duration: float
    processing_time: float
    speed_ratio: float
    model: str
    device: str
    timestamp: str
    status: str = "success"
    engine: str = "sherpa-onnx"


class ONNXModelInfo(BaseModel):
    """ONNX Model information response"""
    id: str
    name: str
    type: str
    language: str
    description: str
    performance_tier: str
    onnx_path: str
    tokens_path: str
    recommended: bool


class ONNXModelListResponse(BaseModel):
    """Response model for available ONNX models"""
    models: List[ONNXModelInfo]
    current_model: Optional[Dict[str, Any]] = None


class ONNXHealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    uptime: float
    onnx_model_loaded: bool
    sherpa_version: Optional[str]
    device: str
    version: str = "1.0.0"


# ONNX Model Configuration
class ONNXModelConfig:
    """Configuration for ONNX models"""
    
    def __init__(self):
        self.models = {
            "whisper-large-v3-onnx": {
                "id": "whisper-large-v3-onnx",
                "name": "Whisper Large-v3 ONNX",
                "type": "onnx_whisper",
                "language": "th",
                "description": "OpenAI Whisper Large-v3 optimized for ONNX runtime with Sherpa-ONNX",
                "performance_tier": "fast",
                "onnx_path": "models/onnx/whisper-large-v3/model.onnx",
                "tokens_path": "models/onnx/whisper-large-v3/tokens.txt",
                "encoder_path": "models/onnx/whisper-large-v3/encoder.onnx", 
                "decoder_path": "models/onnx/whisper-large-v3/decoder.onnx",
                "recommended": True,
                "download_url": "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-large-v3/resolve/main/"
            },
            "whisper-medium-onnx": {
                "id": "whisper-medium-onnx", 
                "name": "Whisper Medium ONNX",
                "type": "onnx_whisper",
                "language": "th",
                "description": "OpenAI Whisper Medium optimized for ONNX runtime",
                "performance_tier": "balanced",
                "onnx_path": "models/onnx/whisper-medium/model.onnx",
                "tokens_path": "models/onnx/whisper-medium/tokens.txt",
                "encoder_path": "models/onnx/whisper-medium/encoder.onnx",
                "decoder_path": "models/onnx/whisper-medium/decoder.onnx", 
                "recommended": False,
                "download_url": "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-medium/resolve/main/"
            }
        }


class SherpaONNXManager:
    """Manages Sherpa-ONNX models for ASR"""
    
    def __init__(self):
        self.current_model = None
        self.current_model_info = None
        self.config = ONNXModelConfig()
        self.recognizer = None
        
        if not SHERPA_AVAILABLE:
            logger.error("❌ Sherpa-ONNX not available. Install with: pip install sherpa-onnx")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available ONNX models"""
        models_list = []
        
        for model_id, model_info in self.config.models.items():
            models_list.append({
                "id": model_id,
                "name": model_info["name"],
                "type": model_info["type"],
                "language": model_info["language"],
                "description": model_info["description"],
                "performance_tier": model_info["performance_tier"],
                "onnx_path": model_info["onnx_path"],
                "tokens_path": model_info["tokens_path"],
                "recommended": model_info["recommended"]
            })
        
        return sorted(models_list, key=lambda x: (not x["recommended"], x["performance_tier"]))
    
    def _download_model_if_needed(self, model_info: Dict[str, Any]) -> bool:
        """Download ONNX model files if they don't exist"""
        try:
            import requests
            
            # Create model directory
            model_dir = Path(model_info["onnx_path"]).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            files_to_download = [
                ("encoder.onnx", model_info["encoder_path"]),
                ("decoder.onnx", model_info["decoder_path"]), 
                ("tokens.txt", model_info["tokens_path"])
            ]
            
            base_url = model_info["download_url"]
            
            for filename, local_path in files_to_download:
                if not os.path.exists(local_path):
                    logger.info(f"📥 Downloading {filename}...")
                    url = f"{base_url}{filename}"
                    
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    logger.info(f"✅ Downloaded {filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download model files: {e}")
            return False
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific ONNX model"""
        if not SHERPA_AVAILABLE:
            raise RuntimeError("Sherpa-ONNX not available")
        
        if model_id not in self.config.models:
            valid = ", ".join(self.config.models.keys())
            raise ValueError(f"Unknown model ID: '{model_id}'. Valid options: {valid}")
        
        model_info = self.config.models[model_id]
        logger.info(f"🔄 Loading ONNX model: {model_info['name']}")
        
        # Download model if needed
        if not self._download_model_if_needed(model_info):
            raise RuntimeError(f"Failed to download model files for {model_id}")
        
        try:
            # Create Sherpa-ONNX configuration
            config = sherpa_onnx.OfflineRecognizerConfig(
                model_config=sherpa_onnx.OfflineModelConfig(
                    whisper=sherpa_onnx.OfflineWhisperModelConfig(
                        encoder=model_info["encoder_path"],
                        decoder=model_info["decoder_path"],
                        language="th",
                        task="transcribe",
                    ),
                    tokens=model_info["tokens_path"],
                    num_threads=4,
                    debug=False,
                    provider="cpu",  # or "cuda" if available
                ),
                lm_config=sherpa_onnx.OfflineLMConfig(),
                feat_config=sherpa_onnx.FeatureConfig(
                    sample_rate=16000,
                    feature_dim=80,
                ),
                decoding_config=sherpa_onnx.OfflineRecognizerConfig.DecodingConfig(
                    method="greedy_search",
                ),
            )
            
            # Create recognizer
            if not sherpa_onnx.OfflineRecognizer.config_validate(config):
                raise RuntimeError("Invalid Sherpa-ONNX configuration")
            
            self.recognizer = sherpa_onnx.OfflineRecognizer(config)
            self.current_model_info = model_info
            
            logger.info(f"✅ ONNX model loaded successfully: {model_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load ONNX model: {e}")
            raise
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using loaded ONNX model"""
        if not self.recognizer:
            raise RuntimeError("No ONNX model loaded")
        
        start_time = time.time()
        
        try:
            # Load and preprocess audio
            audio_data, sample_rate = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                # Simple resampling (for production, use librosa)
                from scipy import signal
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
                sample_rate = 16000
            
            # Ensure float32
            audio_data = audio_data.astype(np.float32)
            
            # Create stream
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio_data)
            
            # Transcribe
            self.recognizer.decode_stream(stream)
            result = self.recognizer.get_result(stream)
            
            processing_time = time.time() - start_time
            duration = len(audio_data) / sample_rate
            
            return {
                "text": result.text.strip(),
                "language": self.current_model_info["language"],
                "duration": duration,
                "processing_time": processing_time,
                "speed_ratio": duration / processing_time if processing_time > 0 else 0,
                "model": self.current_model_info["name"],
                "device": "cpu",  # TODO: detect CUDA
                "engine": "sherpa-onnx"
            }
            
        except Exception as e:
            logger.error(f"❌ ONNX transcription failed: {e}")
            raise
    
    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about currently loaded model"""
        if not self.current_model_info:
            return None
        
        return {
            "id": self.current_model_info["id"],
            "name": self.current_model_info["name"],
            "type": self.current_model_info["type"],
            "language": self.current_model_info["language"],
            "description": self.current_model_info["description"],
            "performance_tier": self.current_model_info["performance_tier"],
            "onnx_path": self.current_model_info["onnx_path"]
        }


# Initialize FastAPI app
app = FastAPI(
    title="ONNX Multi-agent Call Center Backend",
    description="Thai ASR API using Sherpa-ONNX for optimized inference",
    version="1.0.0",
    docs_url="/onnx/docs",
    redoc_url="/onnx/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
onnx_manager: Optional[SherpaONNXManager] = None
start_time = time.time()


# Initialize ONNX model
def initialize_onnx_model(model_id: str = "whisper-large-v3-onnx"):
    """Initialize the ONNX ASR model"""
    global onnx_manager
    
    try:
        logger.info("🚀 Initializing ONNX Model Manager...")
        
        if onnx_manager is None:
            onnx_manager = SherpaONNXManager()
        
        logger.info(f"📦 Loading ONNX model: {model_id}")
        onnx_manager.load_model(model_id)
        
        info = onnx_manager.get_current_model_info() or {}
        logger.info(f"📌 Loaded ONNX model: {info.get('name', 'unknown')}")
        
        logger.info("✅ ONNX ASR model initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ONNX ASR model: {e}")
        onnx_manager = None


# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🎬 Starting ONNX Multi-agent Call Center Backend...")
    
    if not SHERPA_AVAILABLE:
        logger.error("❌ Sherpa-ONNX not available! Install with: pip install sherpa-onnx")
        logger.error("   Server will start but ONNX features will be disabled")
    else:
        initialize_onnx_model()
    
    # Create directories
    os.makedirs("models/onnx", exist_ok=True)
    os.makedirs("audio_uploads", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    logger.info("🎉 ONNX Backend server started successfully!")


@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down ONNX backend server...")


# Health check endpoint
@app.get("/onnx/health", response_model=ONNXHealthResponse)
async def health_check():
    """Health check endpoint for ONNX server"""
    uptime = time.time() - start_time
    
    current_model = onnx_manager.get_current_model_info() if onnx_manager else None
    sherpa_version = getattr(sherpa_onnx, '__version__', None) if SHERPA_AVAILABLE else None
    
    return ONNXHealthResponse(
        status="healthy" if onnx_manager and onnx_manager.recognizer else "degraded",
        timestamp=datetime.now().isoformat(),
        uptime=uptime,
        onnx_model_loaded=onnx_manager is not None and onnx_manager.recognizer is not None,
        sherpa_version=sherpa_version,
        device="cpu"  # TODO: detect CUDA
    )


# Main ONNX ASR endpoint
@app.post("/api/onnx/asr", response_model=ONNXASRResponse)
async def transcribe_audio_onnx(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form("th"),
    model_id: str = Form("whisper-large-v3-onnx"),
):
    """
    Transcribe audio file using Sherpa-ONNX
    
    Args:
        file: Audio file (WAV, MP3, M4A, etc.)
        language: Language code (default: 'th' for Thai)
        model_id: ONNX Model ID to use for transcription
        
    Returns:
        ONNXASRResponse with transcription and metadata
    """
    if not SHERPA_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Sherpa-ONNX not available. Please install: pip install sherpa-onnx"
        )
    
    logger.info(f"/api/onnx/asr called with model_id='{model_id}', language='{language}', file='{file.filename}'")
    
    if onnx_manager is None or onnx_manager.recognizer is None:
        try:
            initialize_onnx_model(model_id)
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="ONNX ASR model not available. Please check server health."
            )
    
    # Check if we need to switch models
    current_model_info = onnx_manager.get_current_model_info()
    if current_model_info and current_model_info.get("id") != model_id:
        logger.info(f"🔄 Switching ONNX model from {current_model_info.get('id')} to {model_id}")
        try:
            onnx_manager.load_model(model_id)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load ONNX model '{model_id}': {str(e)}"
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
        
        # Transcribe audio using ONNX manager
        logger.info("🎵 Starting ONNX transcription...")
        result = onnx_manager.transcribe(temp_file)
        
        logger.info(f"🧾 ONNX transcribed with model='{result.get('model','')}', device='{result.get('device','')}'")
        
        # Create response
        response = ONNXASRResponse(
            text=result["text"],
            language=result["language"],
            duration=result["duration"],
            processing_time=result["processing_time"],
            speed_ratio=result["speed_ratio"],
            model=result["model"],
            device=result["device"],
            timestamp=datetime.now().isoformat(),
            status="success",
            engine=result["engine"]
        )
        
        logger.info(f"✅ ONNX transcription completed: {len(result['text'])} characters")
        return response
        
    except Exception as e:
        logger.error(f"❌ ONNX transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"ONNX transcription failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass


# ONNX Model-related endpoints
@app.get("/api/onnx/models", response_model=ONNXModelListResponse)
async def get_available_onnx_models():
    """Get list of available ONNX models"""
    if onnx_manager is None:
        onnx_manager_instance = SherpaONNXManager()
    else:
        onnx_manager_instance = onnx_manager
    
    available_models = onnx_manager_instance.get_available_models()
    current_model = onnx_manager_instance.get_current_model_info()
    
    return ONNXModelListResponse(
        models=[ONNXModelInfo(**model) for model in available_models],
        current_model=current_model
    )


@app.post("/api/onnx/models/{model_id}/load")
async def load_onnx_model(model_id: str):
    """Load a specific ONNX model"""
    if not SHERPA_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Sherpa-ONNX not available"
        )
    
    if onnx_manager is None:
        raise HTTPException(status_code=503, detail="ONNX model manager not available")
    
    try:
        logger.info(f"/api/onnx/models/{model_id}/load called")
        onnx_manager.load_model(model_id)
        return {
            "status": "success",
            "message": f"ONNX model '{model_id}' loaded successfully",
            "model_info": onnx_manager.get_current_model_info()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load ONNX model: {str(e)}")


# ONNX Model information endpoint
@app.get("/api/onnx/info")
async def get_onnx_asr_info():
    """Get information about the loaded ONNX ASR model"""
    if onnx_manager is None or onnx_manager.recognizer is None:
        raise HTTPException(status_code=503, detail="ONNX ASR model not available")
    
    current_model_info = onnx_manager.get_current_model_info()
    if not current_model_info:
        raise HTTPException(status_code=503, detail="No ONNX model information available")
    
    return current_model_info


# Root endpoint
@app.get("/onnx")
async def onnx_root():
    """Root endpoint for ONNX server"""
    return {
        "message": "ONNX Multi-agent Call Center Backend API",
        "version": "1.0.0",
        "engine": "sherpa-onnx",
        "docs": "/onnx/docs",
        "health": "/onnx/health",
        "endpoints": {
            "onnx_asr": "/api/onnx/asr",
            "onnx_models": "/api/onnx/models",
            "onnx_info": "/api/onnx/info",
            "load_onnx_model": "/api/onnx/models/{model_id}/load"
        },
        "supported_models": [
            "whisper-large-v3-onnx (recommended)",
            "whisper-medium-onnx"
        ],
        "sherpa_available": SHERPA_AVAILABLE
    }


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"❌ Unhandled ONNX exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal ONNX server error: {str(exc)}"}
    )


if __name__ == "__main__":
    # Development server
    logger.info("🚀 Starting ONNX development server...")
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("ONNX_PORT", "8001"))  # Different port
    reload = os.getenv("DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "onnx_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
