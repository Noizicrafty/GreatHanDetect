@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 exit /b 1
)
.venv\Scripts\python.exe -m pip install -r requirements\application.txt
if errorlevel 1 exit /b 1
echo Application installed. Optional cloud codec wheel: install from organiser Jupyter packages folder.
echo Start: START_APPLICATION_WINDOWS.bat
