@echo off
REM MANJU Voice Chatbot Docker Setup Script for Windows
echo 🐳 MANJU Voice Chatbot Docker Setup
echo ==================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first:
    echo    https://docs.docker.com/desktop/windows/
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Desktop first:
    echo    https://docs.docker.com/desktop/windows/
    pause
    exit /b 1
)

echo ✅ Docker and Docker Compose are available
echo.

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist audio_uploads mkdir audio_uploads
if not exist results mkdir results
if not exist temp mkdir temp

REM Build and start services
echo 🔨 Building and starting services...
echo This may take several minutes on first run...
echo.

REM Build the application
docker-compose build

REM Start services
docker-compose up -d

REM Wait for services to be ready
echo.
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check service status
echo.
echo 📊 Service Status:
docker-compose ps

REM Wait for Ollama to download models
echo.
echo 📥 Waiting for Ollama to download phi3 model...
echo This may take several minutes depending on your internet connection...
echo.

REM Monitor for completion (simplified for Windows)
set /a counter=0
:download_loop
docker-compose logs ollama-setup 2>nul | findstr "Phi3 model downloaded successfully" >nul
if %errorlevel% equ 0 (
    echo ✅ Phi3 model downloaded successfully!
    goto :download_complete
)

if %counter% geq 300 (
    echo ⚠️ Model download timeout. You can check manually with:
    echo    docker-compose logs ollama-setup
    goto :download_complete
)

echo    Still downloading... (%counter%/300 seconds^)
timeout /t 10 /nobreak >nul
set /a counter=%counter%+10
goto :download_loop

:download_complete
echo.
echo 🎉 Setup Complete!
echo.
echo 📋 Access Information:
echo    🌐 Voice Chatbot: http://localhost:8501
echo    🧠 Ollama API: http://localhost:11434
echo.
echo 🔧 Useful Commands:
echo    📊 Check status: docker-compose ps
echo    📜 View logs: docker-compose logs -f
echo    🛑 Stop services: docker-compose down
echo    🔄 Restart: docker-compose restart
echo    🗑️ Remove all: docker-compose down -v
echo.
echo 💡 Troubleshooting:
echo    - If Ollama is not responding, wait a few minutes for model download
echo    - Check logs with: docker-compose logs ollama
echo    - Restart Ollama: docker-compose restart ollama
echo.
echo 🎤 Ready to use! Open http://localhost:8501 in your browser
echo.
pause
