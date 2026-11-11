param([int]$Port = 8011)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$trackingPython = Join-Path $PSScriptRoot '.venv-gpu\Scripts\python.exe'
& $trackingPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; x=torch.randn(32,32,device='cuda'); y=x@x; torch.cuda.synchronize(); print('GPU:',torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $trackingPython -m uvicorn legacy_app:app --host localhost --port $Port
exit $LASTEXITCODE
