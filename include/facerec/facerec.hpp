#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace facerec {

struct Match {
  std::string id;
  std::string name;
  float score = 0.f;
};

struct Config {
  std::string yunet_path = "models/face_detection_yunet_2023mar.onnx";
  std::string sface_path = "models/face_recognition_sface_2021dec.onnx";
  std::string gallery_dir = "data/gallery";
  float match_threshold = 0.363f;
  float score_threshold = 0.6f;
  float nms_threshold = 0.3f;
  int top_k = 5000;
  int input_width = 320;
  int input_height = 320;
  int camera_id = 0;
  int camera_warmup_frames = 8;
};

class Engine {
 public:
  explicit Engine(Config cfg);
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) noexcept;
  Engine& operator=(Engine&&) noexcept;

  // Registers the largest face in the image. Returns person id.
  std::string registerFace(const cv::Mat& bgr, const std::string& name);

  // Grab one frame from webcam, then register.
  std::string registerFaceFromCamera(const std::string& name, int camera_id = -1);

  // Interactive preview: SPACE to capture/register, ESC/Q to cancel.
  // Returns person id, or empty string if cancelled.
  std::string registerFaceFromCameraInteractive(const std::string& name, int camera_id = -1);

  // Returns matches sorted by score descending (above threshold only).
  std::vector<Match> recognize(const cv::Mat& bgr, int top_k = 1);

  // Grab one frame from webcam, then recognize.
  std::vector<Match> recognizeFromCamera(int top_k = 1, int camera_id = -1);

  // Interactive live recognition overlay until ESC/Q.
  // Returns the last successful match set (may be empty).
  std::vector<Match> recognizeFromCameraInteractive(int top_k = 1, int camera_id = -1);

  bool removePerson(const std::string& id);

  // Returns (id, name) pairs.
  std::vector<std::pair<std::string, std::string>> listPersons() const;

  const Config& config() const { return cfg_; }

 private:
  int resolveCameraId(int camera_id) const;

  struct Impl;
  Config cfg_;
  Impl* impl_;
};

// Decode image bytes (JPEG/PNG/...) to BGR Mat. Empty on failure.
cv::Mat decodeImage(const std::vector<uchar>& bytes);

// Open webcam, discard warmup frames, return one BGR frame.
cv::Mat captureCameraFrame(int device_id = 0, int warmup_frames = 8);

}  // namespace facerec
