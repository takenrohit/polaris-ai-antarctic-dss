@echo off
REM POLARIS-AI Windows One-Command Startup Script
echo Starting POLARIS-AI Antarctic Decision Support System...
python run_demo.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Demo verification failed.
    exit /b %ERRORLEVEL%
)

echo.
echo Launching Backend and Frontend Services...
start "POLARIS-AI Backend" cmd /k "cd backend && python run_backend.py"
start "POLARIS-AI Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================================
echo Services are starting:
echo   - Backend:  http://127.0.0.1:8000
echo   - Swagger:  http://127.0.0.1:8000/docs
echo   - Frontend: http://localhost:5173
echo ========================================================
