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
from fastapi.responses import JSONResponse, FileResponse
from starlette.responses import Response as StarletteResponse
import json
from pydantic import BaseModel
import uvicorn
import importlib
from starlette.datastructures import UploadFile as StarletteUploadFile

# Multi-agent LLM orchestration
multi_agent_import_error: Optional[str] = None
try:
    from MultiAgent_New import VoiceCallCenterMultiAgent
except Exception as e:
    VoiceCallCenterMultiAgent = None  # Will validate on first use
    multi_agent_import_error = str(e)

# TTS API router
tts_router_import_error: Optional[str] = None
try:
    tts_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "F5-TTS-THAI-API", "src")
    sys.path.append(tts_src_path)
    from f5_tts.f5_api_new_integrate import router as tts_router
except Exception as e:
    tts_router = None
    tts_router_import_error = str(e)

# Attempt to import the TTS module for direct internal calls (optional)
tts_module = None
try:
    tts_module = importlib.import_module("f5_tts.f5_api_new_integrate")
except Exception:
    tts_module = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
start_time = time.time()
model_manager = None  # removed/unused: kept as None for backward-compatibility placeholder
start_time = time.time()
multi_agent: Optional[Any] = None

# Import TyphoonASR wrapper (OOP refactor)
try:
    from typhoon_asr import TyphoonASR
except Exception:
    TyphoonASR = None

# Typhoon ASR instance (primary ASR backend)
typhoon_asr: Optional[Any] = None


# legacy model_manager removed. TyphoonASR is the primary ASR backend.


# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🎬 Starting Multi-agent Call Center ..")
    logger.info(f"🧪 Python executable: {sys.executable}")
    logger.info(f"🧪 OPENROUTER_API_KEY set: {bool(os.getenv('OPENROUTER_API_KEY'))}")
    # Do not eagerly initialize legacy whisper models here. Instead, create TyphoonASR
    # wrapper instance which will lazy-load its heavy model on first transcription.
    # Legacy whisper/model_manager removed - TyphoonASR is primary backend.

    global typhoon_asr
    if TyphoonASR is not None:
        try:
            typhoon_asr = TyphoonASR()
            logger.info("TyphoonASR instance created; model will be lazy-loaded on first use")
            # Preload TyphoonASR at startup on CPU (hard-coded)
            try:
                device = "cpu"
                logger.info("Preloading TyphoonASR model on device='cpu' (this may take a while)")
                typhoon_asr.load_model(device=device)
                logger.info("TyphoonASR model preloaded successfully on CPU")
            except Exception as e:
                logger.warning(f"TyphoonASR preload failed on CPU: {e}")
        except Exception as e:
            typhoon_asr = None
            logger.warning(f"Failed to create TyphoonASR instance: {e}")
    else:
        logger.warning("TyphoonASR class not available; /api/asr will be disabled")
    
    # Create directories if they don't exist
    os.makedirs("audio_uploads", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    logger.info("🎉 Backend server started successfully!")

    # Lazy init VoiceCallCenterMultiAgent so server can start even if CrewAI isn't installed
    global multi_agent
    if VoiceCallCenterMultiAgent is not None:
        try:
            multi_agent = VoiceCallCenterMultiAgent()
            logger.info("🧠 VoiceCallCenterMultiAgent orchestrator initialized")
            logger.info(f"🧠 Multi-agent type: {type(multi_agent).__name__} | model: {multi_agent.config.model}")
        except Exception as e:
            # Improve clarity for missing/incorrect API key errors without leaking secrets
            checked_keys = {
                'OPENROUTER_API_KEY': bool(os.getenv('OPENROUTER_API_KEY')),
                'TOGETHER_API_KEY': bool(os.getenv('TOGETHER_API_KEY')),
                'OPENAI_API_KEY': bool(os.getenv('OPENAI_API_KEY')),
            }
            model_hint = os.getenv('LLM_MODEL', '<default>')
            logger.warning(
                "⚠️ VoiceCallCenterMultiAgent init skipped: %s | checked_keys=%s | model=%s",
                e,
                checked_keys,
                model_hint,
            )
            # Also log full exception traceback at debug level so we can find the upstream source
            logger.exception("VoiceCallCenterMultiAgent initialization exception (full traceback): %s", e)
            logger.debug(
                "If you expected VoiceCallCenterMultiAgent to initialize: set OPENROUTER_API_KEY or TOGETHER_API_KEY,"
                " ensure your LLM_MODEL uses a provider prefix (e.g. 'openrouter/...' or 'together_ai/...'),"
                " or set OPENAI_API_KEY if using OpenAI."
            )
    else:
        if multi_agent_import_error:
            logger.warning(f"⚠️ VoiceCallCenterMultiAgent import failed: {multi_agent_import_error}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down backend server...")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    # Prefer TyphoonASR readiness; fall back to legacy model_manager for info
    current_model = None
    asr_model_loaded = False
    if typhoon_asr is not None and getattr(typhoon_asr, '_model', None) is not None:
        asr_model_loaded = True
        current_model = {"id": getattr(typhoon_asr, 'model_name', 'typhoon'), "device": getattr(typhoon_asr, '_device', 'unknown')}
    else:
        current_model = model_manager.get_current_model_info() if model_manager else None
        asr_model_loaded = bool(current_model)
    
    # LLM health (auto-init to ensure readiness reflected on health)
    llm_ready = False
    llm_model = None
    llm_engine = None
    global multi_agent
    if multi_agent is None and VoiceCallCenterMultiAgent is not None:
        try:
            multi_agent = VoiceCallCenterMultiAgent()
        except Exception as e:
            logger.warning(f"⚠️ LLM init during /health failed: {e}")
    if multi_agent is not None:
        try:
            st = multi_agent.get_system_status()
            llm_ready = bool(st.get("ready", False))
            llm_model = st.get("model")
            llm_engine = st.get("engine")
        except Exception:
            llm_ready = False

    return HealthResponse(
        status="healthy" if asr_model_loaded else "degraded",
        timestamp=datetime.now().isoformat(),
        uptime=uptime,
        asr_model_loaded=asr_model_loaded,
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
        if VoiceCallCenterMultiAgent is None:
            raise HTTPException(
                status_code=503,
                detail=f"VoiceCallCenterMultiAgent unavailable. Install crewai and litellm. Cause: {multi_agent_import_error}",
            )
        try:
            multi_agent = VoiceCallCenterMultiAgent()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed initializing VoiceCallCenterMultiAgent: {e}")

    try:
        st = multi_agent.get_system_status()
        return {
            "engine": st.get("engine"),
            "model": st.get("model"),
            "base_url": st.get("base_url"),
            "tools": st.get("tools"),
            "mock_products_count": st.get("mock_products_count"),
            "ready": st.get("ready"),
            "architecture": st.get("architecture"),
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
        if VoiceCallCenterMultiAgent is None:
            raise HTTPException(
                status_code=503,
                detail=f"VoiceCallCenterMultiAgent unavailable. Install crewai and litellm, then restart server. Cause: {multi_agent_import_error}",
            )
        try:
            multi_agent = VoiceCallCenterMultiAgent()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed initializing VoiceCallCenterMultiAgent: {e}")

    try:
        logger.info("/llm called; generating response via VoiceCallCenterMultiAgent")
        result = multi_agent.process_voice_input(req.text, conversation_history=req.history)
        return LLMResponse(
            response=result.get("response", ""),
            model=result.get("model"),
            used_base_url=None,  # Not returned by process_voice_input
            timestamp=datetime.now().isoformat(),
            status="success",
        )
    except Exception as e:
        logger.error(f"❌ LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")


# Main ASR endpoint
@app.post("/api/asr")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    forward_to_llm: bool = Form(False, description="If true, forward the transcription to /llm and return the LLM result"),
    synthesize: bool = Form(False, description="If true and forward_to_llm=True, synthesize the LLM result via TTS"),
    fast_mode_tts: bool = Form(False, description="Enable fast TTS mode when synthesizing"),
    return_tts_file: bool = Form(False, description="Return TTS file response directly if TTS used"),
    history: Optional[str] = Form(None, description="Optional conversation history JSON string (list of {role,content})")
):
    """
    Transcribe audio file to Thai text using selected model
    
    Args:
        file: Audio file (WAV, MP3, M4A, etc.)
        language: Language code (default: 'th' for Thai)
        model_id: Model ID to use for transcription

        
    Returns:
        ASRResponse with transcription and metadata
    """
    # Use TyphoonASR as primary transcription backend
    global typhoon_asr
    if typhoon_asr is None:
        raise HTTPException(status_code=503, detail="ASR backend not available")
    
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
        
        # Transcribe with TyphoonASR (it will prepare audio and load model lazily)
        logger.info("🎵 Starting transcription with TyphoonASR...")
        result = typhoon_asr.transcribe_file(temp_file)

        # If requested, forward the transcript to the LLM and (optionally) TTS
        if forward_to_llm:
            # Create a new UploadFile backed by the temporary file we've saved
            try:
                new_handle = open(temp_file, 'rb')
                new_upload = StarletteUploadFile(new_handle, filename=Path(file.filename).name)
                resp = await transcribe_and_forward_to_llm(
                    file=new_upload,
                    synthesize=synthesize,
                    fast_mode_tts=fast_mode_tts,
                    return_tts_file=return_tts_file,
                    history=history,
                )
                try:
                    new_handle.close()
                except Exception:
                    pass
                return resp
            except Exception as e:
                logger.error(f"Failed to create upload for forward_to_llm: {e}")
                raise HTTPException(status_code=500, detail=f"Forward-to-LLM failed: {e}")

        return result
        
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


@app.post("/api/asr/llm")
async def transcribe_and_forward_to_llm(
    file: UploadFile = File(..., description="Audio file to transcribe and send to LLM"),
    synthesize: bool = Form(False, description="If true, synthesize the LLM response via /api/tts/tts using the uploaded (reference) audio"),
    fast_mode_tts: bool = Form(False, description="If true, enable fast_mode for TTS generation if available"),
    return_tts_file: bool = Form(False, description="If true, return the generated TTS file path in the response"),
    history: Optional[str] = Form(None, description="Optional conversation history JSON string (list of {role,content})")
):
    """Endpoint: Accept audio, transcribe via TyphoonASR, forward text to LLM, optionally synthesize LLM response via F5-TTS.

    Steps:
    - Save uploaded file to temporary disk
    - Transcribe with TyphoonASR
    - Lazy init and call multi_agent.process_voice_input(transcribed_text, conversation_history)
    - Optionally call f5_tts.f5_api_new_integrate.text_to_speech with the original audio as reference and the LLM response as gen_text
    """
    global typhoon_asr, multi_agent, tts_module

    if typhoon_asr is None:
        raise HTTPException(status_code=503, detail="ASR backend not available")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}")

    temp_file = None
    try:
        audio_data = await file.read()
        logger.info(f"📁 Received audio for /api/asr/llm: {file.filename} ({len(audio_data)} bytes)")

        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(audio_data)
            temp_file = tmp.name

        # Transcribe
        logger.info("🎵 Transcribing audio (ASR -> LLM flow)")
        asr_result = typhoon_asr.transcribe_file(temp_file)
        transcript = asr_result.get('text') if isinstance(asr_result, dict) else str(asr_result)

        # Prepare conversation_history if provided
        conversation_history_obj = None
        if history:
            try:
                conversation_history_obj = json.loads(history)
            except Exception:
                conversation_history_obj = None

        # Ensure LLM orchestrator
        if multi_agent is None:
            if VoiceCallCenterMultiAgent is None:
                raise HTTPException(status_code=503, detail=f"VoiceCallCenterMultiAgent unavailable. Cause: {multi_agent_import_error}")
            try:
                multi_agent = VoiceCallCenterMultiAgent()
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Failed initializing VoiceCallCenterMultiAgent: {e}")

        # Call multi-agent
        try:
            logger.info("🧠 Forwarding transcript to LLM: %s", transcript)
            llm_result = multi_agent.process_voice_input(transcript, conversation_history=conversation_history_obj)
        except Exception as e:
            logger.error(f"❌ LLM processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

        response_payload = {
            'transcription': transcript,
            'asr_meta': asr_result,
            'llm_response': llm_result,
        }

        # Optionally synthesize the LLM response via TTS module
        if synthesize:
            if tts_module is None:
                logger.warning("⚠️ TTS module not available; cannot synthesize LLM response.")
                response_payload['tts'] = {'success': False, 'reason': 'TTS module not available'}
            else:
                # Create a starlette UploadFile using temp_file
                try:
                    upload_handle = open(temp_file, 'rb')
                    ref_upload = StarletteUploadFile(upload_handle, filename=Path(temp_file).name)
                    # Call text_to_speech directly (it is async)
                    logger.info("🔊 Calling TTS to synthesize LLM response")
                    tts_resp = await tts_module.text_to_speech(
                        ref_audio=ref_upload,
                        ref_text=transcript,
                        gen_text=llm_result.get('response', '') if isinstance(llm_result, dict) else str(llm_result),
                        remove_silence=True,
                        nfe_step=8 if fast_mode_tts else 16,
                        cfg_strength=2.0,
                        seed=-1,
                        fast_mode=fast_mode_tts,
                        return_file=return_tts_file,
                    )
                    # tts_resp may be a FileResponse (if return_file True) or a dict/pydantic model
                    # If the TTS layer returned a FileResponse, return it directly to the caller.
                    if isinstance(tts_resp, StarletteResponse):
                        # Close the ref_upload handle before returning
                        try:
                            upload_handle.close()
                        except Exception:
                            pass
                        return tts_resp
                    response_payload['tts'] = {'success': True, 'result': tts_resp}
                except Exception as e:
                    logger.error(f"❌ TTS synth failed: {e}")
                    response_payload['tts'] = {'success': False, 'reason': str(e)}
                finally:
                    try:
                        upload_handle.close()
                    except Exception:
                        pass

        return response_payload

    except (KeyboardInterrupt, SystemExit) as e:
        logger.error(f"🛑 Server interrupt during /api/asr/llm processing: {e}")
        raise HTTPException(status_code=500, detail="Server interrupted during processing")
    except Exception as e:
        logger.error(f"❌ /api/asr/llm failed: {e}")
        raise HTTPException(status_code=500, detail=f"ASR->LLM flow failed: {str(e)}")
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

@app.get("/api/models", response_model=ModelListResponse)
async def get_available_models():
    """Get list of available ASR models"""
    # This server uses TyphoonASR as its ASR backend. The legacy model manager
    # was removed. We return the TyphoonASR model name if loaded, otherwise empty list.
    models = []
    current_model = None
    if typhoon_asr is not None:
        current_model = {"id": getattr(typhoon_asr, 'model_name', 'typhoon'), "name": getattr(typhoon_asr, 'model_name', 'typhoon'), "type": "typhoon-asr", "language": "th", "description": "Typhoon ASR model", "performance_tier": "unknown", "recommended": True}

    return ModelListResponse(models=models, current_model=current_model)

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
            "asr_llm": "/api/asr/llm",
            "asr_forward": "/api/asr?forward_to_llm=true",
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


# Mount TTS router if available
if tts_router:
    app.include_router(tts_router, prefix="/api/tts", tags=["TTS"])
    logger.info("✅ TTS API router mounted at /api/tts")
elif tts_router_import_error:
    logger.warning(f"⚠️ TTS API router not available: {tts_router_import_error}")


if __name__ == "__main__":
    # Development server
    logger.info("🚀 Starting development server...")
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "new_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
