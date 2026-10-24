@echo off
cd /d "%~dp0"
if not exist .venv-demo\Scripts\python.exe (
  py -3.12 -m venv .venv-demo
  if errorlevel 1 exit /b 1
)
.venv-demo\Scripts\python.exe -c "import reportlab,openpyxl,PIL" >nul 2>&1
if errorlevel 1 (
  .venv-demo\Scripts\python.exe -m pip install -r requirements\demo.txt
  if errorlevel 1 exit /b 1
)
echo Open http://127.0.0.1:8090/
if "%~1"=="" (
  .venv-demo\Scripts\python.exe -c "import urllib.request,json; assert json.load(urllib.request.urlopen('http://127.0.0.1:8090/api/demo/state',timeout=2))['demo'] is True" >nul 2>&1
  if not errorlevel 1 (
    start "" "http://127.0.0.1:8090/"
    exit /b 0
  )
)
.venv-demo\Scripts\python.exe tools\run_presentation_demo.py %*
