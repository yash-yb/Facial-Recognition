#!/usr/bin/env bash
# Download YuNet + SFace ONNX models (Hugging Face OpenCV Zoo mirrors).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"

download() {
  local url="$1" out="$2" min="$3"
  local dest="$MODELS/$out"
  if [[ -f "$dest" ]]; then
    local size
    size=$(wc -c < "$dest" | tr -d ' ')
    if [[ "$size" -ge "$min" ]]; then
      echo "Already present: $out ($size bytes)"
      return
    fi
  fi
  echo "Downloading $out..."
  curl -L --fail "$url" -o "$dest"
  local size
  size=$(wc -c < "$dest" | tr -d ' ')
  if [[ "$size" -lt "$min" ]]; then
    echo "error: $out looks too small ($size bytes)" >&2
    exit 1
  fi
  echo "OK $out ($size bytes)"
}

download \
  "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx" \
  "face_detection_yunet_2023mar.onnx" \
  100000

download \
  "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx" \
  "face_recognition_sface_2021dec.onnx" \
  1000000

echo "Models saved to $MODELS"
