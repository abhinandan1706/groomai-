@echo off
echo ========================================
echo   GroomAI - Development Server
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

:: Set development environment variables
set FLASK_ENV=development
set FLASK_DEBUG=true
set PORT=5000

:: Create uploads directory if it doesn't exist
if not exist "uploads" mkdir uploads

:: Start the development server
echo.
echo Starting GroomAI development server...
echo Debug mode: ENABLED
echo Auto-reload: ENABLED
echo Open your browser and navigate to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app_enhanced.py

pause
