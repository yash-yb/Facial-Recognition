#include "facerec/facerec.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>

namespace {

void usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " register --name <name> (--image <path> | --camera [id]) [options]\n"
      << "  " << argv0 << " recognize (--image <path> | --camera [id]) [--top-k N] [options]\n"
      << "  " << argv0 << " list [options]\n"
      << "  " << argv0 << " remove --id <id> [options]\n"
      << "\nCamera:\n"
      << "  --camera [id]      Use webcam (default id 0). Interactive preview in CLI.\n"
      << "  SPACE / S         Capture during register\n"
      << "  ESC / Q           Quit camera preview\n"
      << "\nOptions:\n"
      << "  --yunet <path>     YuNet ONNX (default models/face_detection_yunet_2023mar.onnx)\n"
      << "  --sface <path>     SFace ONNX (default models/face_recognition_sface_2021dec.onnx)\n"
      << "  --gallery <dir>    Gallery directory (default data/gallery)\n"
      << "  --threshold <f>    Cosine match threshold (default 0.363)\n";
}

std::string requireArg(int& i, int argc, char** argv, const char* flag) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + flag);
  }
  return argv[++i];
}

// --camera optionally takes an integer id as the next argv token.
bool parseCameraFlag(int& i, int argc, char** argv, bool& use_camera, int& camera_id) {
  use_camera = true;
  if (i + 1 < argc) {
    const std::string next = argv[i + 1];
    if (!next.empty() && (next[0] != '-' || (next.size() > 1 && next[1] >= '0' && next[1] <= '9'))) {
      // Numeric device id (including negative via explicit next parse).
      bool numeric = true;
      std::size_t start = 0;
      if (next[0] == '-' && next.size() > 1) {
        start = 1;
      }
      for (std::size_t c = start; c < next.size(); ++c) {
        if (next[c] < '0' || next[c] > '9') {
          numeric = false;
          break;
        }
      }
      if (numeric && !(next[0] == '-' && next.size() == 1)) {
        camera_id = std::stoi(next);
        ++i;
      }
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      usage(argv[0]);
      return 1;
    }

    const std::string cmd = argv[1];
    facerec::Config cfg;
    std::string name;
    std::string image_path;
    std::string id;
    int top_k = 1;
    bool use_camera = false;
    int camera_id = 0;

    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--name") {
        name = requireArg(i, argc, argv, "--name");
      } else if (a == "--image") {
        image_path = requireArg(i, argc, argv, "--image");
      } else if (a == "--id") {
        id = requireArg(i, argc, argv, "--id");
      } else if (a == "--camera") {
        parseCameraFlag(i, argc, argv, use_camera, camera_id);
      } else if (a == "--yunet") {
        cfg.yunet_path = requireArg(i, argc, argv, "--yunet");
      } else if (a == "--sface") {
        cfg.sface_path = requireArg(i, argc, argv, "--sface");
      } else if (a == "--gallery") {
        cfg.gallery_dir = requireArg(i, argc, argv, "--gallery");
      } else if (a == "--threshold") {
        cfg.match_threshold = std::stof(requireArg(i, argc, argv, "--threshold"));
      } else if (a == "--top-k") {
        top_k = std::stoi(requireArg(i, argc, argv, "--top-k"));
      } else if (a == "-h" || a == "--help") {
        usage(argv[0]);
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + a);
      }
    }

    if (use_camera) {
      cfg.camera_id = camera_id;
    }

    facerec::Engine engine(cfg);

    if (cmd == "register") {
      if (name.empty()) {
        throw std::runtime_error("register requires --name");
      }
      if (use_camera && !image_path.empty()) {
        throw std::runtime_error("register: use either --image or --camera, not both");
      }
      if (!use_camera && image_path.empty()) {
        throw std::runtime_error("register requires --image <path> or --camera [id]");
      }

      if (use_camera) {
        const std::string pid = engine.registerFaceFromCameraInteractive(name, camera_id);
        if (pid.empty()) {
          std::cerr << "cancelled\n";
          return 1;
        }
        std::cout << "registered id=" << pid << " name=" << name << "\n";
        return 0;
      }

      const cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
      if (img.empty()) {
        throw std::runtime_error("failed to read image: " + image_path);
      }
      const std::string pid = engine.registerFace(img, name);
      std::cout << "registered id=" << pid << " name=" << name << "\n";
      return 0;
    }

    if (cmd == "recognize") {
      if (use_camera && !image_path.empty()) {
        throw std::runtime_error("recognize: use either --image or --camera, not both");
      }
      if (!use_camera && image_path.empty()) {
        throw std::runtime_error("recognize requires --image <path> or --camera [id]");
      }

      if (use_camera) {
        const auto matches = engine.recognizeFromCameraInteractive(top_k, camera_id);
        if (matches.empty()) {
          std::cout << "no match\n";
          return 0;
        }
        for (const auto& m : matches) {
          std::cout << "id=" << m.id << " name=" << m.name << " score=" << m.score << "\n";
        }
        return 0;
      }

      const cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
      if (img.empty()) {
        throw std::runtime_error("failed to read image: " + image_path);
      }
      const auto matches = engine.recognize(img, top_k);
      if (matches.empty()) {
        std::cout << "no match\n";
        return 0;
      }
      for (const auto& m : matches) {
        std::cout << "id=" << m.id << " name=" << m.name << " score=" << m.score << "\n";
      }
      return 0;
    }

    if (cmd == "list") {
      for (const auto& p : engine.listPersons()) {
        std::cout << "id=" << p.first << " name=" << p.second << "\n";
      }
      return 0;
    }

    if (cmd == "remove") {
      if (id.empty()) {
        throw std::runtime_error("remove requires --id");
      }
      if (!engine.removePerson(id)) {
        std::cerr << "not found: " << id << "\n";
        return 1;
      }
      std::cout << "removed " << id << "\n";
      return 0;
    }

    usage(argv[0]);
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
