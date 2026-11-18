@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo   Starting Grounding DINO Object Counting Web Dashboard
echo ========================================================

REM Find python executable
if exist "..\Object_Counting\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\Object_Counting\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python: %PYTHON_EXE%
%PYTHON_EXE% web_app.py --port 5000

pause
