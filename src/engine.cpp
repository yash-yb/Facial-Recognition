#include "facerec/facerec.hpp"

#include "gallery.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

namespace facerec {
namespace {

void l2Normalize(std::vector<float>& v) {
  double sum = 0.0;
  for (float x : v) {
    sum += static_cast<double>(x) * static_cast<double>(x);
  }
  const double norm = std::sqrt(sum);
  if (norm < 1e-12) {
    throw std::runtime_error("cannot normalize zero embedding");
  }
  const float inv = static_cast<float>(1.0 / norm);
  for (float& x : v) {
    x *= inv;
  }
}

int largestFaceIndex(const cv::Mat& faces) {
  int best = 0;
  float best_area = -1.f;
  for (int i = 0; i < faces.rows; ++i) {
    const float w = faces.at<float>(i, 2);
    const float h = faces.at<float>(i, 3);
    const float area = w * h;
    if (area > best_area) {
      best_area = area;
      best = i;
    }
  }
  return best;
}

}  // namespace

struct Engine::Impl {
  cv::Ptr<cv::FaceDetectorYN> detector;
  cv::Ptr<cv::FaceRecognizerSF> recognizer;
  Gallery gallery;
  cv::Mat aligned;
  cv::Mat feature;

  explicit Impl(const Config& cfg)
      : gallery(cfg.gallery_dir) {
    detector = cv::FaceDetectorYN::create(
        cfg.yunet_path, "",
        cv::Size(cfg.input_width, cfg.input_height),
        cfg.score_threshold, cfg.nms_threshold, cfg.top_k);
    if (!detector) {
      throw std::runtime_error("failed to load YuNet model: " + cfg.yunet_path);
    }

    recognizer = cv::FaceRecognizerSF::create(cfg.sface_path, "");
    if (!recognizer) {
      throw std::runtime_error("failed to load SFace model: " + cfg.sface_path);
    }

    gallery.load();
  }

  cv::Mat detectLargest(const cv::Mat& bgr) {
    if (bgr.empty()) {
      throw std::runtime_error("empty image");
    }
    detector->setInputSize(bgr.size());
    cv::Mat faces;
    detector->detect(bgr, faces);
    if (faces.empty() || faces.rows < 1) {
      throw std::runtime_error("no face detected");
    }
    const int idx = largestFaceIndex(faces);
    return faces.row(idx).clone();
  }

  std::vector<float> embedFace(const cv::Mat& bgr, const cv::Mat& face_row) {
    recognizer->alignCrop(bgr, face_row, aligned);
    recognizer->feature(aligned, feature);
    if (feature.empty()) {
      throw std::runtime_error("failed to extract face feature");
    }
    cv::Mat flat = feature.reshape(1, 1);
    std::vector<float> emb(flat.begin<float>(), flat.end<float>());
    l2Normalize(emb);
    return emb;
  }
};

Engine::Engine(Config cfg) : cfg_(std::move(cfg)), impl_(new Impl(cfg_)) {}

Engine::~Engine() { delete impl_; }

Engine::Engine(Engine&& other) noexcept : cfg_(std::move(other.cfg_)), impl_(other.impl_) {
  other.impl_ = nullptr;
}

Engine& Engine::operator=(Engine&& other) noexcept {
  if (this != &other) {
    delete impl_;
    cfg_ = std::move(other.cfg_);
    impl_ = other.impl_;
    other.impl_ = nullptr;
  }
  return *this;
}

std::string Engine::registerFace(const cv::Mat& bgr, const std::string& name) {
  const cv::Mat face = impl_->detectLargest(bgr);
  const auto emb = impl_->embedFace(bgr, face);
  return impl_->gallery.add(name, emb);
}

std::vector<Match> Engine::recognize(const cv::Mat& bgr, int top_k) {
  const cv::Mat face = impl_->detectLargest(bgr);
  const auto emb = impl_->embedFace(bgr, face);
  const auto hits = impl_->gallery.match(emb, top_k, cfg_.match_threshold);
  std::vector<Match> out;
  out.reserve(hits.size());
  for (const auto& h : hits) {
    Match m;
    m.id = h.first.id;
    m.name = h.first.name;
    m.score = h.second;
    out.push_back(std::move(m));
  }
  return out;
}

bool Engine::removePerson(const std::string& id) { return impl_->gallery.remove(id); }

std::vector<std::pair<std::string, std::string>> Engine::listPersons() const {
  return impl_->gallery.list();
}

cv::Mat decodeImage(const std::vector<uchar>& bytes) {
  if (bytes.empty()) {
    return {};
  }
  return cv::imdecode(bytes, cv::IMREAD_COLOR);
}

}  // namespace facerec
