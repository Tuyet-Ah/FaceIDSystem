@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. See README.md for installation instructions.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8000 in your browser.
echo Loaded models: YOLO11 (pc_v7), LOCA (Few-Shot/Zero-Shot), Grounding DINO (Prompt), and AnomalyCLIP.
echo Video processing: Background job with H.264 export (YOLO11 & AnomalyCLIP).
echo Press Ctrl+C in this window to stop the web app.
".venv\Scripts\python.exe" web_app.py --weights models/pc_v7_best.pt --imgsz 960
pause
