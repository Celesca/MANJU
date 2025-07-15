@echo off
echo Installing F5-TTS-THAI for premium Thai text-to-speech...
echo.

echo Step 1: Installing PyTorch and torchaudio...
echo This may take a few minutes depending on your internet connection...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
if %ERRORLEVEL% NEQ 0 (
    echo Warning: CUDA version installation failed, trying CPU version...
    pip install torch torchaudio
)

echo.
echo Step 2: Installing additional dependencies...
pip install soundfile librosa numpy

echo.
echo Step 3: Installing F5-TTS-THAI from GitHub...
echo This may take a while for the first time...
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: F5-TTS-THAI installation failed.
    echo Please check your internet connection and try again.
    echo You can also try manual installation:
    echo   git clone https://github.com/VYNCX/F5-TTS-THAI.git
    echo   cd F5-TTS-THAI
    echo   pip install .
    pause
    exit /b 1
)

echo.
echo Step 4: Testing installation...
python test_f5_tts.py

echo.
echo Installation complete!
echo.
echo F5-TTS-THAI Features:
echo - Highest quality Thai TTS
echo - Natural sounding voice
echo - Works offline
echo - Customizable voice cloning
echo.
echo Note: F5-TTS works best with CUDA GPU acceleration
echo Check GPU availability with: python -c "import torch; print(torch.cuda.is_available())"
echo.
echo If you encounter issues:
echo 1. Run test_f5_tts.py to diagnose problems
echo 2. Check the debug panel in the chatbot sidebar
echo 3. F5-TTS will automatically fall back to gTTS if needed
echo.
pause
