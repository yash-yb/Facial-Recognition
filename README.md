# Facial Recognition (C++ / OpenCV)

Optimized face registration and recognition library using **OpenCV YuNet** (detection) and **SFace** (embeddings), with an optional HTTP API for use from any project.

The previous Python / MediaPipe prototype path is retired; this tree is the supported implementation.

## Features

- **Register** a face → L2-normalized embedding stored on disk
- **Recognize** a face → cosine similarity (dot product) against the gallery
- **C++ library** (`libfacerec`) for in-process use
- **HTTP server** (`facerec_server`) for language-agnostic clients

## Layout

| Path | Description |
|------|-------------|
| `include/facerec/facerec.hpp` | Public C++ API |
| `src/` | Engine + embedding gallery |
| `server/` | Thin REST wrapper |
| `examples/cli.cpp` | File-based CLI |
| `models/` | YuNet + SFace ONNX (download) |
| `data/gallery/` | Persisted embeddings |

## Dependencies

- CMake ≥ 3.16
- C++17 compiler
- OpenCV 4.5.4+ with `core`, `imgproc`, `imgcodecs`, `dnn`, `objdetect` (YuNet / SFace APIs)

### Windows (MSYS2 UCRT64)

```bash
pacman -S mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-opencv mingw-w64-ucrt-x86_64-toolchain
```

## Download models

Models are fetched from the official OpenCV Zoo mirrors on Hugging Face:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1
```

Linux / macOS:

```bash
bash scripts/download_models.sh
```

Expected files:

- `models/face_detection_yunet_2023mar.onnx`
- `models/face_recognition_sface_2021dec.onnx`

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

On Windows with Visual Studio generators, binaries land under `build/Release/`. With Ninja/Makefiles: `build/facerec_cli`, `build/facerec_server`.

## CLI

```bash
# Register from image
./facerec_cli register --name Alice --image alice.jpg

# Register from webcam (preview window: SPACE = capture, ESC/Q = cancel)
./facerec_cli register --name Alice --camera
./facerec_cli register --name Alice --camera 1   # second camera

# Recognize from image
./facerec_cli recognize --image unknown.jpg

# Recognize from webcam (live overlay until ESC/Q)
./facerec_cli recognize --camera

# List / remove
./facerec_cli list
./facerec_cli remove --id <id>
```

## HTTP API

```bash
./facerec_server --host 0.0.0.0 --port 8080
```

| Method | Path | Body |
|--------|------|------|
| `GET` | `/health` | — |
| `POST` | `/v1/register` | `{"name":"Alice","image_base64":"..."}` **or** `{"name":"Alice","from_camera":true,"camera_id":0}` |
| `POST` | `/v1/recognize` | `{"image_base64":"...","top_k":1}` **or** `{"from_camera":true,"camera_id":0,"top_k":1}` |
| `GET` | `/v1/faces` | — |
| `DELETE` | `/v1/faces/{id}` | — |

`from_camera` grabs a frame from the **server machine’s** webcam (not the client’s).

### curl examples

```bash
# Health
curl http://127.0.0.1:8080/health

# Register (camera on server)
curl -s http://127.0.0.1:8080/v1/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Alice","from_camera":true}'

# Recognize (camera on server)
curl -s http://127.0.0.1:8080/v1/recognize \
  -H "Content-Type: application/json" \
  -d '{"from_camera":true,"top_k":1}'

# List
curl -s http://127.0.0.1:8080/v1/faces
```

PowerShell register (image):

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("alice.jpg"))
$body = @{ name = "Alice"; image_base64 = $b64 } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/register -Method Post -Body $body -ContentType "application/json"
```

PowerShell register (server webcam):

```powershell
$body = @{ name = "Alice"; from_camera = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/register -Method Post -Body $body -ContentType "application/json"
```

## C++ library usage

```cpp
#include "facerec/facerec.hpp"
#include <opencv2/imgcodecs.hpp>

int main() {
  facerec::Config cfg;
  facerec::Engine engine(cfg);

  // From image
  cv::Mat img = cv::imread("alice.jpg");
  std::string id = engine.registerFace(img, "Alice");

  // One-shot webcam grab
  id = engine.registerFaceFromCamera("Bob");
  auto matches = engine.recognizeFromCamera(/*top_k=*/1);

  // Interactive preview windows
  engine.registerFaceFromCameraInteractive("Carol");
  engine.recognizeFromCameraInteractive();
}
```

Link against `facerec` and OpenCV.

## Notes

- When multiple faces are present, the **largest** face is used.
- Default cosine threshold is `0.363` (OpenCV SFace ballpark); tune with `--threshold`.
- Gallery files: `data/gallery/index.json` + `data/gallery/embeddings/<id>.bin`.
- CUDA / OpenVINO backends are not required for v1; OpenCV CPU DNN is the default.
