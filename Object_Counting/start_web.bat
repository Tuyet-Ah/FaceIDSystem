@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. See README.md for installation instructions.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8000 in your browser.
echo Running with high-resolution imgsz 960 for detailed detection.
echo Press Ctrl+C in this window to stop the web app.
".venv\Scripts\python.exe" web_app.py --imgsz 960
pause
