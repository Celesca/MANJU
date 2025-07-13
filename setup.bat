@echo off
echo Installing Voice Chatbot Dependencies...
echo.

echo Step 1: Uninstalling any existing PyTorch installation...
pip uninstall torch torchvision torchaudio -y

echo.
echo Step 2: Installing PyTorch with proper CPU support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Step 3: Installing other Python packages...
pip install -r requirements.txt

echo.
echo Step 4: Checking Ollama installation...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ollama is not installed. Please install it from: https://ollama.ai
    echo After installation, run: ollama pull phi3
) else (
    echo Ollama is installed!
    echo Pulling phi3 model...
    ollama pull phi3
)

echo.
echo Step 5: Installation complete!
echo.
echo To run the voice chatbot:
echo 1. Start Ollama server: ollama serve
echo 2. Run the chatbot: streamlit run voice_chatbot.py
echo.
pause
