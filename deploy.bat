@echo off
echo ===============================================
echo       GroomAI Production Deployment
echo ===============================================
echo.

:: Set color for better visibility
color 0A

:: Check if virtual environment exists
if not exist "tf-env\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please ensure tf-env directory exists.
    pause
    exit /b 1
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call tf-env\Scripts\activate.bat

:: Check if model file exists
if not exist "model\groomai_skin_model.h5" (
    echo [ERROR] Model file not found!
    echo Please ensure model\groomai_skin_model.h5 exists.
    pause
    exit /b 1
)

:: Create necessary directories
echo [INFO] Creating necessary directories...
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs

:: Set production environment variables
echo [INFO] Setting production environment variables...
set FLASK_ENV=production
set FLASK_DEBUG=false
set PORT=5000
set PYTHONPATH=%CD%

:: Install/upgrade production dependencies
echo [INFO] Installing production dependencies...
tf-env\Scripts\pip.exe install --upgrade gunicorn waitress

:: Create gunicorn configuration
echo [INFO] Creating production server configuration...

:: Test model loading first
echo [INFO] Testing model loading...
tf-env\Scripts\python.exe -c "from model_loader import load_groomai_model; model = load_groomai_model(r'C:\Users\Administrator\Downloads\GroomAI-model\model\groomai_skin_model.h5'); print('✅ Model loaded successfully!' if model else '❌ Model failed to load')"

if errorlevel 1 (
    echo [ERROR] Model loading test failed!
    pause
    exit /b 1
)

:: Run health check
echo [INFO] Running application health check...
start /B tf-env\Scripts\python.exe -c "
import sys
sys.path.append('.')
from app_enhanced import create_app
app = create_app()
with app.test_client() as client:
    response = client.get('/health')
    if response.status_code == 200:
        print('✅ Health check passed!')
    else:
        print('❌ Health check failed!')
        sys.exit(1)
"

timeout /t 3 >nul

echo.
echo ===============================================
echo           Deployment Options
echo ===============================================
echo 1. Start Development Server (Flask built-in)
echo 2. Start Production Server (Gunicorn - Recommended)
echo 3. Start Windows Production Server (Waitress)
echo 4. Exit
echo.
set /p choice="Please select an option (1-4): "

if "%choice%"=="1" goto dev_server
if "%choice%"=="2" goto gunicorn_server
if "%choice%"=="3" goto waitress_server
if "%choice%"=="4" goto end

:dev_server
echo.
echo [INFO] Starting Flask Development Server...
echo [WARNING] This is for development only!
echo Server will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
tf-env\Scripts\python.exe app_enhanced.py
goto end

:gunicorn_server
echo.
echo [INFO] Starting Gunicorn Production Server...
echo Server will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
tf-env\Scripts\gunicorn.exe --config gunicorn.conf.py app_enhanced:app
goto end

:waitress_server
echo.
echo [INFO] Starting Waitress Production Server (Windows optimized)...
echo Server will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
tf-env\Scripts\waitress-serve.exe --host=0.0.0.0 --port=5000 --call app_enhanced:create_app
goto end

:end
echo.
echo [INFO] Deployment script completed.
pause
