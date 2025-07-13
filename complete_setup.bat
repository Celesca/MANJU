@echo off
echo Complete Voice Chatbot Setup - Handles All Issues
echo ================================================
echo.

echo Step 1: Installing FFmpeg...
echo Trying winget installation...
winget install "FFmpeg (Essentials Build)" --accept-source-agreements --accept-package-agreements >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ FFmpeg installed via winget
) else (
    echo ⚠️ Winget failed, trying chocolatey...
    choco install ffmpeg -y >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ FFmpeg installed via chocolatey
    ) else (
        echo ❌ Automatic FFmpeg installation failed
        echo Manual installation required - see instructions below
    )
)

echo.
echo Step 2: Cleaning conflicting packages...
pip uninstall torch torchvision torchaudio tensorflow keras tf-keras -y

echo.
echo Step 3: Installing compatibility packages...
pip install tf-keras

echo.
echo Step 4: Installing PyTorch CPU...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Step 5: Installing other requirements...
pip install -r requirements.txt

echo.
echo Step 6: Testing installations...
python -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>nul || echo "❌ PyTorch failed"
python -c "import transformers; print('✅ Transformers: OK')" 2>nul || echo "❌ Transformers failed"
python -c "import streamlit; print('✅ Streamlit: OK')" 2>nul || echo "❌ Streamlit failed"

echo.
echo Step 7: Testing FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ FFmpeg is working
) else (
    echo ❌ FFmpeg not found in PATH
    echo.
    echo 📥 Manual FFmpeg Installation Required:
    echo 1. Download: https://github.com/BtbN/FFmpeg-Builds/releases
    echo 2. Look for: ffmpeg-master-latest-win64-gpl.zip
    echo 3. Extract to: C:\ffmpeg
    echo 4. Add to PATH: C:\ffmpeg\bin
    echo 5. Restart computer
    echo.
    echo Alternative - Portable Installation:
    echo 1. Download from: https://www.gyan.dev/ffmpeg/builds/
    echo 2. Extract anywhere
    echo 3. Set FFMPEG_BINARY env var to ffmpeg.exe path
)

echo.
echo Step 8: Checking Ollama...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ollama not installed
    echo Download from: https://ollama.ai
    echo Then run: ollama pull phi3
) else (
    echo ✅ Ollama found
    echo Pulling phi3 model...
    ollama pull phi3
)

echo.
echo ================================================
echo Setup Summary:
echo ================================================
python -c "
try:
    import torch
    print('✅ PyTorch: OK')
except: print('❌ PyTorch: Failed')

try:
    import transformers
    print('✅ Transformers: OK')
except: print('❌ Transformers: Failed')

try:
    import streamlit
    print('✅ Streamlit: OK')
except: print('❌ Streamlit: Failed')

try:
    import pydub
    print('✅ PyDub: OK')
except: print('❌ PyDub: Failed')
"

echo.
echo 🎯 Next Steps:
echo 1. If FFmpeg failed, follow manual installation above
echo 2. Restart your computer if you installed FFmpeg
echo 3. Run: streamlit run voice_chatbot.py
echo.
echo 🔧 If issues persist:
echo 1. Try: streamlit run simple_chatbot.py (text-only)
echo 2. Run: python fix_all_issues.py
echo 3. Install Visual C++ Redistributable if needed
echo.
pause
