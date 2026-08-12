#include "facerec/facerec.hpp"

#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace facerec {
namespace {

cv::VideoCapture openCamera(int device_id) {
  cv::VideoCapture cap;
#if defined(_WIN32)
  cap.open(device_id, cv::CAP_DSHOW);
  if (!cap.isOpened()) {
    cap.open(device_id);
  }
#else
  cap.open(device_id);
#endif
  if (!cap.isOpened()) {
    throw std::runtime_error("failed to open camera id=" + std::to_string(device_id));
  }
  return cap;
}

void drawHud(cv::Mat& frame, const std::string& line1, const std::string& line2 = {}) {
  cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 56), cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(frame, line1, cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1,
              cv::LINE_AA);
  if (!line2.empty()) {
    cv::putText(frame, line2, cv::Point(10, 46), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1,
                cv::LINE_AA);
  }
}

}  // namespace

cv::Mat captureCameraFrame(int device_id, int warmup_frames) {
  cv::VideoCapture cap = openCamera(device_id);
  cv::Mat frame;
  const int warm = warmup_frames < 0 ? 0 : warmup_frames;
  for (int i = 0; i < warm; ++i) {
    cap.grab();
  }
  if (!cap.read(frame) || frame.empty()) {
    throw std::runtime_error("failed to capture frame from camera id=" + std::to_string(device_id));
  }
  return frame;
}

int Engine::resolveCameraId(int camera_id) const {
  return camera_id < 0 ? cfg_.camera_id : camera_id;
}

std::string Engine::registerFaceFromCamera(const std::string& name, int camera_id) {
  const int id = resolveCameraId(camera_id);
  const cv::Mat frame = captureCameraFrame(id, cfg_.camera_warmup_frames);
  return registerFace(frame, name);
}

std::string Engine::registerFaceFromCameraInteractive(const std::string& name, int camera_id) {
  const int id = resolveCameraId(camera_id);
  cv::VideoCapture cap = openCamera(id);
  const std::string win = "facerec register - SPACE capture, ESC/Q cancel";
  cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

  cv::Mat frame;
  std::string person_id;
  while (true) {
    if (!cap.read(frame) || frame.empty()) {
      cv::destroyWindow(win);
      throw std::runtime_error("camera read failed");
    }
    cv::Mat view = frame.clone();
    drawHud(view, "Register: " + name, "SPACE = capture | ESC/Q = cancel");
    cv::imshow(win, view);
    const int key = cv::waitKey(1) & 0xFF;
    if (key == 27 || key == 'q' || key == 'Q') {
      break;
    }
    if (key == ' ' || key == 's' || key == 'S') {
      try {
        person_id = registerFace(frame, name);
        drawHud(view, "Registered OK", person_id);
        cv::imshow(win, view);
        cv::waitKey(800);
        break;
      } catch (const std::exception& ex) {
        drawHud(view, "Capture failed", ex.what());
        cv::imshow(win, view);
        cv::waitKey(1000);
      }
    }
  }
  cv::destroyWindow(win);
  return person_id;
}

std::vector<Match> Engine::recognizeFromCamera(int top_k, int camera_id) {
  const int id = resolveCameraId(camera_id);
  const cv::Mat frame = captureCameraFrame(id, cfg_.camera_warmup_frames);
  return recognize(frame, top_k);
}

std::vector<Match> Engine::recognizeFromCameraInteractive(int top_k, int camera_id) {
  const int id = resolveCameraId(camera_id);
  cv::VideoCapture cap = openCamera(id);
  const std::string win = "facerec recognize - ESC/Q quit";
  cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

  cv::Mat frame;
  std::vector<Match> last;
  while (true) {
    if (!cap.read(frame) || frame.empty()) {
      cv::destroyWindow(win);
      throw std::runtime_error("camera read failed");
    }

    cv::Mat view = frame.clone();
    std::string status = "no match";
    try {
      last = recognize(frame, top_k);
      if (!last.empty()) {
        status = last.front().name + " (" + cv::format("%.3f", last.front().score) + ")";
      }
    } catch (const std::exception&) {
      status = "no face";
      last.clear();
    }

    drawHud(view, "Recognize (live)", status + " | ESC/Q = quit");
    cv::imshow(win, view);
    const int key = cv::waitKey(1) & 0xFF;
    if (key == 27 || key == 'q' || key == 'Q') {
      break;
    }
  }
  cv::destroyWindow(win);
  return last;
}

}  // namespace facerec
