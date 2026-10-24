@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python tools\bootstrap_source.py
if errorlevel 1 goto fail
python tools\run_host_baseline.py --video 0 --device cpu --show
:fail
pause
