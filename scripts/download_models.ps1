# Download YuNet + SFace ONNX models from OpenCV Zoo (Git LFS).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ModelsDir = Join-Path $Root "models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$Tmp = Join-Path $env:TEMP ("opencv_zoo_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

try {
  Write-Host "Cloning opencv_zoo (sparse, LFS)..."
  Push-Location $Tmp
  git clone --depth 1 --filter=blob:none --sparse https://github.com/opencv/opencv_zoo.git repo
  Set-Location repo
  git sparse-checkout set models/face_detection_yunet models/face_recognition_sface

  $gitLfs = Get-Command git-lfs -ErrorAction SilentlyContinue
  if ($gitLfs) {
    git lfs install
    git lfs pull
  } else {
    Write-Host "git-lfs not found; attempting media.githubusercontent.com download..."
    $files = @(
      @{
        Rel = "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        Out = "face_detection_yunet_2023mar.onnx"
      },
      @{
        Rel = "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        Out = "face_recognition_sface_2021dec.onnx"
      }
    )
    foreach ($f in $files) {
      $url = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/$($f.Rel)"
      $dest = Join-Path $ModelsDir $f.Out
      Write-Host "Downloading $($f.Out)..."
      Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    }
    Pop-Location
    Write-Host "Models saved to $ModelsDir"
    return
  }

  $srcYunet = Join-Path (Get-Location) "models\face_detection_yunet\face_detection_yunet_2023mar.onnx"
  $srcSface = Join-Path (Get-Location) "models\face_recognition_sface\face_recognition_sface_2021dec.onnx"
  Copy-Item -Force $srcYunet (Join-Path $ModelsDir "face_detection_yunet_2023mar.onnx")
  Copy-Item -Force $srcSface (Join-Path $ModelsDir "face_recognition_sface_2021dec.onnx")
  Pop-Location

  foreach ($name in @("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx")) {
    $p = Join-Path $ModelsDir $name
    $len = (Get-Item $p).Length
    if ($len -lt 10000) {
      throw "Downloaded file looks too small (possible LFS pointer): $p ($len bytes). Install git-lfs and re-run."
    }
    Write-Host "OK $name ($len bytes)"
  }
  Write-Host "Models saved to $ModelsDir"
}
finally {
  if (Test-Path $Tmp) {
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
  }
}
