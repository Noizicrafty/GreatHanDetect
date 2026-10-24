@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup_application_windows.bat first.
  exit /b 2
)
.venv\Scripts\python.exe -m greathan %*
