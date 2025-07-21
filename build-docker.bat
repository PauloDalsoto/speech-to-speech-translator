@echo off
echo 🐳 Building Speech-to-Speech Translator Docker Image...

REM Build the Docker image
docker build -f .dockerfile -t speech-to-speech-translator:latest .

if %ERRORLEVEL% EQU 0 (
    echo ✅ Docker image built successfully!
    echo.
    echo 🚀 You can now run the container with:
    echo    docker run -it --rm speech-to-speech-translator:latest
    echo.
    echo Or use docker-compose:
    echo    docker-compose up
    echo.
    echo 📝 Don't forget to:
    echo    1. Copy config_example.json to config.json
    echo    2. Configure your API keys in config.json
    echo    3. Adjust audio settings as needed
) else (
    echo ❌ Docker build failed!
    exit /b %ERRORLEVEL%
)

pause
