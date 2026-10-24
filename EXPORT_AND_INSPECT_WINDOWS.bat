@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python tools\bootstrap_source.py
if errorlevel 1 goto fail
python tools\export_onnx.py --part both --device cpu
if errorlevel 1 goto fail
python tools\inspect_onnx.py --part detector
if errorlevel 1 goto fail
python tools\inspect_onnx.py --part pose
if errorlevel 1 goto fail
echo Export and validation OK (TPU compile requires Ubuntu SDK).
pause
exit /b 0
:fail
echo Export failed.
pause
exit /b 1
