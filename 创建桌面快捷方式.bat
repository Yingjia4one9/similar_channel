@echo off
chcp 65001 >nul
echo ========================================
echo   创建桌面快捷方式
echo ========================================
echo.

:: 检查 start.bat 是否存在
if not exist "%~dp0start.bat" (
    echo 错误: 找不到 start.bat 文件！
    echo 请确保在项目根目录下运行此脚本。
    pause
    exit /b 1
)

:: 使用 VBScript 创建快捷方式（更稳定，支持中文）
cscript //nologo "%~dp0create_shortcut.vbs"

echo.
echo 快捷方式已创建到桌面！
echo.
pause
