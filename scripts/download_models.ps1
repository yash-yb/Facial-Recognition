# Download YuNet + SFace ONNX models (Hugging Face OpenCV Zoo mirrors).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ModelsDir = Join-Path $Root "models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$files = @(
  @{
    Url = "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
    Out = "face_detection_yunet_2023mar.onnx"
    MinBytes = 100000
  },
  @{
    Url = "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"
    Out = "face_recognition_sface_2021dec.onnx"
    MinBytes = 1000000
  }
)

foreach ($f in $files) {
  $dest = Join-Path $ModelsDir $f.Out
  if ((Test-Path $dest) -and ((Get-Item $dest).Length -ge $f.MinBytes)) {
    Write-Host "Already present: $($f.Out) ($((Get-Item $dest).Length) bytes)"
    continue
  }
  Write-Host "Downloading $($f.Out)..."
  Invoke-WebRequest -Uri $f.Url -OutFile $dest -UseBasicParsing
  $len = (Get-Item $dest).Length
  if ($len -lt $f.MinBytes) {
    throw "Downloaded file looks too small: $dest ($len bytes)"
  }
  Write-Host "OK $($f.Out) ($len bytes)"
}

Write-Host "Models saved to $ModelsDir"
