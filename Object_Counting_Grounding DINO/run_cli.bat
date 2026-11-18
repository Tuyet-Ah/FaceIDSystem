@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo   Running Grounding DINO Object Counting (CLI)
echo ========================================================

REM Find python executable
if exist "..\Object_Counting\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\Object_Counting\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

%PYTHON_EXE% count.py --prompt "computer tower. pc case." %*

pause
