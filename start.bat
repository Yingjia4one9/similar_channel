@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo   YouTube Similar Channel Service Starting...
echo ========================================
echo.
echo Service URL: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the service
echo ========================================
echo.
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
