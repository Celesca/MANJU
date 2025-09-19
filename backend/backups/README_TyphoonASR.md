TyphoonASR Backend (server)

This project now uses the TyphoonASR wrapper as the primary ASR backend. The legacy `whisper` model_manager and faster_whisper code were retained in the repo but are no longer used by `/api/asr` endpoints.

Quick start

1. Install Python dependencies (examples):

```powershell
python -m pip install -r backend\requirements.txt
# If you don't have a requirements.txt, install minimal packages:
python -m pip install librosa soundfile
# NeMo/ASR and PyTorch are required for the Typhoon model; follow NeMo installation docs for your CUDA/PyTorch version.
```

2. Run the server (from repo root):

```powershell
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

Notes
- `TyphoonASR` lazy-loads the model on first transcription; startup will be fast but the first transcription will include model load time.
- If you want to preload the model at startup, edit `backend/server.py` and call `typhoon_asr.load_model(device='cpu' or 'cuda')` inside the `startup_event()`.
- `model_id` parameter in `/api/asr` is preserved for compatibility but does not change the Typhoon model used. Implement mapping if you need to select variants.

Endpoints
- POST /api/asr (multipart file) -> transcribe a single file
- POST /api/asr/batch -> transcribe up to 10 files
- GET /health -> server + ASR readiness
- POST /api/asr/reload -> recreate TyphoonASR instance (forces reload on next use)

Troubleshooting
- If `TyphoonASR` fails to import, ensure `backend/typhoon_asr.py` is present and dependencies (librosa, soundfile, NeMo) are installed.
- For GPU support, ensure PyTorch with CUDA is installed and NeMo is installed with matching versions.
