$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$trackingPython = Join-Path $PSScriptRoot '..\BE\venv\Scripts\python.exe'
& $trackingPython legacy_app.py
exit $LASTEXITCODE
