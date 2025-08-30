@echo off
echo 🚀 Starting Multi-agent Call Center Backend Server
echo ================================================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Check if we're in the right directory
if not exist "server.py" (
    echo ❌ server.py not found in current directory
    echo 💡 Please run this script from the backend directory
    pause
    exit /b 1
)

:: Check if requirements are installed
echo 📦 Checking dependencies...
python -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ FastAPI dependencies not found
    echo 📥 Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

:: Check if faster-whisper is installed
echo 🔍 Checking faster-whisper...
python -c "import faster_whisper" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ faster-whisper not found
    echo 📥 Installing faster-whisper...
    pip install faster-whisper
    if errorlevel 1 (
        echo ❌ Failed to install faster-whisper
        pause
        exit /b 1
    )
)

:: Start the server
echo ✅ All dependencies ready
echo 🎬 Starting server...
echo.
echo 📋 Server will be available at:
echo    🌐 http://localhost:8000
echo    📚 API Docs: http://localhost:8000/docs
echo    🔍 Health: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

python server.py

pause
