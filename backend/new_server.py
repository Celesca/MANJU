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
    # Also try to import the module so we can call a synth function directly if available
    try:
        import importlib, inspect
        f5_api_module = importlib.import_module("f5_tts.f5_api_new_integrate")
        # Try to find a plausible synth callable on the module or the router's routes
        f5_synthesize_fn = None
        for candidate in ("synthesize_text", "synthesize", "synthesize_tts", "tts_synthesize", "generate_tts", "speak"):
            if hasattr(f5_api_module, candidate):
                f5_synthesize_fn = getattr(f5_api_module, candidate)
                break
        # If not found on module, try to inspect router endpoints for a matching POST handler
        if f5_synthesize_fn is None and hasattr(tts_router, "routes"):
            for r in tts_router.routes:
                try:
                    path = getattr(r, "path", "")
                except Exception:
                    path = ""
                if "synth" in path or "tts" in path or "speak" in path or "generate" in path:
                    # endpoint may be wrapped; try to use r.endpoint if callable
                    endpoint = getattr(r, "endpoint", None)
                    if callable(endpoint):
                        f5_synthesize_fn = endpoint
                        break
    except Exception as e:
        f5_api_module = None
        f5_synthesize_fn = None
        tts_router_import_error = str(e)
except Exception as e:
    tts_router = None
    tts_router_import_error = str(e)

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


@app.post("/api/talk")
async def talk_pipe(
    file: UploadFile = File(..., description="Audio file to process"),
    tts: bool = Form(False, description="Whether to synthesize the LLM response to speech using F5 TTS"),
    background_tasks: BackgroundTasks = None,
):
    """
    Full pipeline endpoint: ASR -> LLM -> (optional) F5 TTS

    - Accepts an audio file
    - Transcribes via TyphoonASR
    - Sends transcription to VoiceCallCenterMultiAgent to generate a response
    - Optionally synthesizes the response via F5 TTS (if available)
    """
    global typhoon_asr, multi_agent, f5_synthesize_fn, tts_router

    if typhoon_asr is None:
        raise HTTPException(status_code=503, detail="ASR backend not available")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    temp_file = None
    try:
        audio_data = await file.read()
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix.lower(), delete=False) as tmp:
            tmp.write(audio_data)
            temp_file = tmp.name

        # Run transcription in thread to avoid blocking
        logger.info("/api/talk: starting transcription")
        try:
            trans_result = await asyncio.to_thread(typhoon_asr.transcribe_file, temp_file)
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

        # Normalize transcription text
        if isinstance(trans_result, dict):
            asr_text = trans_result.get("text") or trans_result.get("transcription") or trans_result.get("transcript") or str(trans_result)
        else:
            asr_text = str(trans_result)

        # Ensure LLM orchestrator exists
        if multi_agent is None:
            if VoiceCallCenterMultiAgent is None:
                logger.warning("VoiceCallCenterMultiAgent unavailable for /api/talk")
                raise HTTPException(status_code=503, detail=f"LLM orchestrator unavailable: {multi_agent_import_error}")
            try:
                multi_agent = VoiceCallCenterMultiAgent()
            except Exception as e:
                logger.error(f"Failed to init multi_agent: {e}")
                raise HTTPException(status_code=503, detail=f"LLM init failed: {e}")

        # Call LLM (may be sync); run in thread
        logger.info("/api/talk: sending transcription to multi-agent LLM")
        try:
            llm_res = await asyncio.to_thread(multi_agent.process_voice_input, asr_text, None)
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"LLM processing failed: {e}")

        if isinstance(llm_res, dict):
            llm_text = llm_res.get("response") or llm_res.get("text") or str(llm_res)
        else:
            llm_text = str(llm_res)

        tts_info = None
        # Optionally synthesize with F5 TTS
        if tts:
            logger.info("/api/talk: attempting TTS synthesis via F5")
            try:
                synth_result = None
                # If we discovered a synth callable, try to call it
                if 'f5_synthesize_fn' in globals() and f5_synthesize_fn:
                    try:
                        import inspect
                        fn = f5_synthesize_fn
                        sig = None
                        try:
                            sig = inspect.signature(fn)
                        except Exception:
                            sig = None

                        kwargs = {}
                        # Favor common parameter names
                        if sig is not None:
                            params = list(sig.parameters.keys())
                            if 'text' in params:
                                kwargs['text'] = llm_text
                            elif 'input_text' in params:
                                kwargs['input_text'] = llm_text
                            elif len(params) >= 1:
                                # pass as first positional arg by calling without kwargs
                                pass

                        if inspect.iscoroutinefunction(fn):
                            if kwargs:
                                synth_result = await fn(**kwargs)
                            else:
                                synth_result = await fn(llm_text)
                        else:
                            if kwargs:
                                synth_result = await asyncio.to_thread(fn, **kwargs)
                            else:
                                synth_result = await asyncio.to_thread(fn, llm_text)

                        tts_info = {"status": "synthesized", "result": synth_result}
                    except Exception as e:
                        logger.warning(f"F5 synth callable failed: {e}")
                        tts_info = {"status": "failed", "error": str(e)}
                else:
                    # Fallback: try to find a POST route on tts_router and call its endpoint
                    if tts_router is not None:
                        synth_called = False
                        for r in tts_router.routes:
                            try:
                                path = getattr(r, 'path', '')
                            except Exception:
                                path = ''
                            if 'synth' in path or 'tts' in path or 'speak' in path or 'generate' in path:
                                endpoint = getattr(r, 'endpoint', None)
                                if callable(endpoint):
                                    try:
                                        import inspect
                                        if inspect.iscoroutinefunction(endpoint):
                                            synth_result = await endpoint(text=llm_text)
                                        else:
                                            synth_result = await asyncio.to_thread(endpoint, text=llm_text)
                                        tts_info = {"status": "synthesized_via_router", "result": synth_result, "route": path}
                                        synth_called = True
                                        break
                                    except Exception as e:
                                        logger.warning(f"Router endpoint call failed: {e}")
                                        tts_info = {"status": "failed", "error": str(e)}
                        if not synth_called and tts_info is None:
                            tts_info = {"status": "not_available", "detail": "No F5 synth callable or router endpoint found"}
                    else:
                        tts_info = {"status": "not_available", "detail": "F5 TTS router not mounted"}
            except Exception as e:
                logger.error(f"Unexpected error during TTS: {e}")
                tts_info = {"status": "failed", "error": str(e)}

        return {
            "asr": asr_text,
            "llm": llm_text,
            "tts": tts_info,
        }

    except (KeyboardInterrupt, SystemExit) as e:
        logger.error(f"Server interrupt during /api/talk: {e}")
        raise HTTPException(status_code=500, detail="Server interrupted during processing")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected /api/talk error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
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
