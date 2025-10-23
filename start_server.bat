@echo off
echo ========================================
echo    GroomAI - AI Skin Analysis Server
echo ========================================
echo.

:: Check if virtual environment exists
if not exist "tf-env\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_environment.bat first
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call tf-env\Scripts\activate.bat

:: Set environment variables
set FLASK_ENV=production
set FLASK_DEBUG=false
set PORT=5000

:: Check if model file exists
if not exist "model\groomai_skin_model.h5" (
    echo ERROR: Model file not found!
    echo Please ensure model\groomai_skin_model.h5 exists
    pause
    exit /b 1
)

:: Create uploads directory if it doesn't exist
if not exist "uploads" mkdir uploads

:: Start the server
echo.
echo Starting GroomAI server...
echo Open your browser and navigate to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app_enhanced.py

pause
