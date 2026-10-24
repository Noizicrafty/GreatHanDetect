@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 goto fail
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements\export.txt
if errorlevel 1 goto fail
python tools\doctor.py
pause
exit /b 0
:fail
echo ERROR: setup failed.
pause
exit /b 1
