@echo off
REM Start Thai ASR Real-time WebSocket Server
REM This script starts the real-time Thai ASR WebSocket server with GPU optimization

echo ==========================================
echo Thai ASR Real-time WebSocket Server
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Check if we're in the correct directory
if not exist "realtime_thai_asr_server.py" (
    echo ERROR: realtime_thai_asr_server.py not found
    echo Please run this script from the backend directory
    pause
    exit /b 1
)

echo Checking Python dependencies...
python -c "import websockets, numpy, scipy" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Some dependencies might be missing
    echo Installing requirements...
    pip install -r requirements_realtime.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Starting Thai ASR WebSocket Server...
echo.
echo WebSocket Endpoints:
echo   Control: ws://localhost:8765
echo   Audio:   ws://localhost:8766
echo.
echo Web Client: realtime_thai_asr_client.html
echo.
echo Press Ctrl+C to stop the server
echo ==========================================
echo.

REM Start the server
python realtime_thai_asr_server.py

echo.
echo Server stopped.
pause
