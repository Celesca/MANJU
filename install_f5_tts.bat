@echo off
echo Installing F5-TTS-THAI for premium Thai text-to-speech...
echo.

echo Step 1: Installing PyTorch and torchaudio...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Step 2: Installing F5-TTS-THAI from GitHub...
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git

echo.
echo Step 3: Installing additional dependencies...
pip install soundfile librosa

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
pause
