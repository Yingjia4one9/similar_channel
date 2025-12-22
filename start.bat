@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo   YouTube 相似频道后端服务启动中...
echo ========================================
echo.
echo 服务地址: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 可停止服务
echo ========================================
echo.
python -m uvicorn main:app --host 0.0.0.0 --port 8000
pause

