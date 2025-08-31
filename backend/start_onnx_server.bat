@echo off
echo Starting ONNX Server on port 8001...
cd /d "%~dp0"
python onnx_server.py
pause
