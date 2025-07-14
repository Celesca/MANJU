@echo off
echo Installing F5-TTS-THAI for better Thai Text-to-Speech...
echo.

echo Step 1: Installing PyTorch with CUDA support...
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

echo.
echo Step 2: Installing F5-TTS-THAI...
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git

echo.
echo Installation complete!
echo You can now use F5-TTS-THAI for high-quality Thai text-to-speech.
echo.
pause
