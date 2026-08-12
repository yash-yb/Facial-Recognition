#include "facerec/facerec.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>

namespace {

void usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " register --name <name> --image <path> [options]\n"
      << "  " << argv0 << " recognize --image <path> [--top-k N] [options]\n"
      << "  " << argv0 << " list [options]\n"
      << "  " << argv0 << " remove --id <id> [options]\n"
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

    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--name") {
        name = requireArg(i, argc, argv, "--name");
      } else if (a == "--image") {
        image_path = requireArg(i, argc, argv, "--image");
      } else if (a == "--id") {
        id = requireArg(i, argc, argv, "--id");
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

    facerec::Engine engine(cfg);

    if (cmd == "register") {
      if (name.empty() || image_path.empty()) {
        throw std::runtime_error("register requires --name and --image");
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
      if (image_path.empty()) {
        throw std::runtime_error("recognize requires --image");
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
