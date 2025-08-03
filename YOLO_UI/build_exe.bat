@echo off
echo =================================
echo     YOLO训练工具打包脚本
echo =================================
echo.

:: 检查是否安装了PyInstaller
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo 错误: 未安装PyInstaller
    echo 正在安装PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo 安装PyInstaller失败，请手动安装: pip install pyinstaller
        pause
        exit /b 1
    )
)

echo 开始清理之前的构建文件...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "__pycache__" rmdir /s /q "__pycache__"

echo 开始打包...
echo 这可能需要几分钟时间，请耐心等待...
echo.

:: 使用spec文件进行打包
pyinstaller yolo_ui.spec --clean

if errorlevel 1 (
    echo.
    echo 打包失败！请检查错误信息。
    echo.
    echo 常见问题解决方案:
    echo 1. 确保所有依赖都已安装: pip install -r requirements.txt
    echo 2. 确保在YOLO_UI目录下运行此脚本
    echo 3. 检查防病毒软件是否阻止了打包过程
    pause
    exit /b 1
)

echo.
echo =================================
echo         打包完成！
echo =================================
echo.
echo 可执行文件位置: dist\YOLO_Training_Tool\YOLO_Training_Tool.exe
echo.
echo 注意事项:
echo 1. 首次运行可能需要较长时间初始化
echo 2. 确保目标机器有足够的空间（建议至少2GB可用空间）
echo 3. 如果在其他机器运行出现问题，可能需要安装Visual C++ Redistributable
echo.

pause