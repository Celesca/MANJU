@echo off
echo Advanced Voice Chatbot Setup with GPU Detection...
echo.

echo Step 1: Cleaning existing PyTorch installation...
pip uninstall torch torchvision torchaudio -y

echo.
echo Step 2: Detecting GPU support...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected! Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo No NVIDIA GPU detected. Installing CPU-only PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Step 3: Fixing TensorFlow/Keras compatibility issues...
pip uninstall tensorflow keras tf-keras -y
pip install tf-keras

echo.
echo Step 4: Installing other dependencies...
pip install -r requirements.txt

echo.
echo Step 5: Testing PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || (
    echo PyTorch installation failed! Trying alternative installation...
    pip install torch torchvision torchaudio
)

echo.
echo Step 6: Installing Microsoft Visual C++ Redistributable (if needed)...
echo If you encounter DLL errors, please download and install:
echo https://aka.ms/vs/17/release/vc_redist.x64.exe

echo.
echo Step 7: Checking Ollama installation...
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
echo Step 8: Setup complete!
echo.
echo IMPORTANT: If you still get DLL errors, please:
echo 1. Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo 2. Restart your computer
echo 3. Try running: python -c "import torch; print('PyTorch works!')"
echo.
echo To run the voice chatbot:
echo 1. Start Ollama server: ollama serve
echo 2. Run the chatbot: streamlit run voice_chatbot.py
echo.
pause
