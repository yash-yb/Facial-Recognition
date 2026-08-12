#!/usr/bin/env bash
# Download YuNet + SFace ONNX models from OpenCV Zoo (Git LFS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"
TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

echo "Cloning opencv_zoo (sparse, LFS)..."
cd "$TMP"
git clone --depth 1 --filter=blob:none --sparse https://github.com/opencv/opencv_zoo.git repo
cd repo
git sparse-checkout set models/face_detection_yunet models/face_recognition_sface

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install
  git lfs pull
else
  echo "git-lfs not found; trying media.githubusercontent.com..."
  curl -L "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    -o "$MODELS/face_detection_yunet_2023mar.onnx"
  curl -L "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
    -o "$MODELS/face_recognition_sface_2021dec.onnx"
fi

if [[ -f models/face_detection_yunet/face_detection_yunet_2023mar.onnx ]]; then
  cp -f models/face_detection_yunet/face_detection_yunet_2023mar.onnx "$MODELS/"
  cp -f models/face_recognition_sface/face_recognition_sface_2021dec.onnx "$MODELS/"
fi

for f in face_detection_yunet_2023mar.onnx face_recognition_sface_2021dec.onnx; do
  size=$(wc -c < "$MODELS/$f" | tr -d ' ')
  if [[ "$size" -lt 10000 ]]; then
    echo "error: $f looks like an LFS pointer ($size bytes). Install git-lfs and re-run." >&2
    exit 1
  fi
  echo "OK $f ($size bytes)"
done
echo "Models saved to $MODELS"
