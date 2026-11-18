param(
    [ValidateSet('buffalo_l','buffalo_s','adaface','ghostfacenet')]
    [string]$Model = 'buffalo_l',
    [int]$CameraIndex = 0,
    [switch]$RecordAttendance
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:FACEID_BACKEND = 'insightface'
$env:FACEID_MODEL_NAME = 'buffalo_l'
$env:FACEID_LOCAL_MODEL = $Model
$env:FACEID_LOCAL_RECORD = if ($RecordAttendance) { '1' } else { '0' }
$env:FACEID_ENABLE_LOCAL_CAMERA = '1'
$env:FACEID_SHOW_CAMERA = '1'
$env:FACEID_CAMERA_INDEX = "$CameraIndex"
$env:FACEID_DATABASE = Join-Path $PSScriptRoot 'BE/backend/dtb.db'
$env:FACEID_DATASET_DIR = Join-Path $PSScriptRoot 'BE/dataset'
$env:FACEID_EMBEDDINGS_PATH = Join-Path $PSScriptRoot 'BE/backend/employee_face_embeddings.pkl'
$env:FACEID_ADAFACE_REPO = Join-Path $PSScriptRoot 'BE/backend/AdaFace'
$env:FACEID_ADAFACE_ARCH = 'ir_50'
$env:FACEID_ADAFACE_CKPT = Join-Path $env:FACEID_ADAFACE_REPO 'pretrained/adaface_ir50_webface4m.ckpt'
$env:FACEID_GHOSTFACENET_MODEL = Join-Path $PSScriptRoot 'BE/backend/models/ghostfacenet_v1_w1.3_s1_arcface_ms1mv3.onnx'
$env:FACEID_HOST = '127.0.0.1'
$env:FACEID_PORT = '5000'
Write-Host 'Local camera: click camera window, then 1/2/3/4 to select model. Q closes camera; Ctrl+C stops server.'
& "$PSScriptRoot/BE/venv/Scripts/python.exe" -u "$PSScriptRoot/BE/backend/server.py"
exit $LASTEXITCODE
