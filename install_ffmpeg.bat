@echo off
echo Installing FFmpeg for Audio Processing...
echo.

echo Method 1: Trying winget installation...
winget install "FFmpeg (Essentials Build)" --accept-source-agreements --accept-package-agreements 2>nul
if %errorlevel% equ 0 (
    echo ✅ FFmpeg installed successfully via winget!
    goto :test
)

echo Method 2: Trying chocolatey installation...
choco install ffmpeg -y 2>nul
if %errorlevel% equ 0 (
    echo ✅ FFmpeg installed successfully via chocolatey!
    goto :test
)

echo.
echo ❌ Automatic installation failed. Manual installation required:
echo.
echo 📥 Manual Installation Steps:
echo 1. Download FFmpeg from: https://github.com/BtbN/FFmpeg-Builds/releases
echo 2. Look for "ffmpeg-master-latest-win64-gpl.zip"
echo 3. Extract to C:\ffmpeg
echo 4. Add C:\ffmpeg\bin to your PATH:
echo    - Press Win + R, type "sysdm.cpl", press Enter
echo    - Click "Environment Variables"
echo    - Under "System Variables", find "Path", click "Edit"
echo    - Click "New" and add: C:\ffmpeg\bin
echo    - Click OK on all dialogs
echo 5. Restart your computer
echo.
echo Alternative: Use portable version
echo 1. Download from: https://www.gyan.dev/ffmpeg/builds/
echo 2. Extract anywhere
echo 3. Copy the path to ffmpeg.exe
echo 4. Set FFMPEG_BINARY environment variable to that path
echo.
goto :end

:test
echo.
echo Testing FFmpeg installation...
ffmpeg -version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ FFmpeg is working correctly!
    echo.
    echo You can now run the voice chatbot:
    echo streamlit run voice_chatbot.py
) else (
    echo ❌ FFmpeg installation verification failed.
    echo Please restart your computer and try again.
)

:end
echo.
pause
