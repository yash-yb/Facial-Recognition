#include "facerec/facerec.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

using json = nlohmann::json;

namespace {

std::vector<uchar> base64Decode(const std::string& input) {
  static const int8_t kDecode[256] = {
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,
      52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -2, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,
      7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,
      -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
      49, 50, 51, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

  std::string s = input;
  // Strip data URL prefix if present.
  const auto comma = s.find(',');
  if (s.rfind("data:", 0) == 0 && comma != std::string::npos) {
    s = s.substr(comma + 1);
  }

  std::vector<uchar> out;
  out.reserve(s.size() * 3 / 4);
  int val = 0;
  int valb = -8;
  for (unsigned char c : s) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ') {
      if (c == '=') {
        break;
      }
      continue;
    }
    const int8_t d = kDecode[c];
    if (d < 0) {
      throw std::runtime_error("invalid base64");
    }
    val = (val << 6) + d;
    valb += 6;
    if (valb >= 0) {
      out.push_back(static_cast<uchar>((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return out;
}

std::string requireArg(int& i, int argc, char** argv, const char* flag) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + flag);
  }
  return argv[++i];
}

void printUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --host <addr>      default 0.0.0.0\n"
      << "  --port <n>         default 8080\n"
      << "  --yunet <path>\n"
      << "  --sface <path>\n"
      << "  --gallery <dir>\n"
      << "  --threshold <f>\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    facerec::Config cfg;
    std::string host = "0.0.0.0";
    int port = 8080;

    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--host") {
        host = requireArg(i, argc, argv, "--host");
      } else if (a == "--port") {
        port = std::stoi(requireArg(i, argc, argv, "--port"));
      } else if (a == "--yunet") {
        cfg.yunet_path = requireArg(i, argc, argv, "--yunet");
      } else if (a == "--sface") {
        cfg.sface_path = requireArg(i, argc, argv, "--sface");
      } else if (a == "--gallery") {
        cfg.gallery_dir = requireArg(i, argc, argv, "--gallery");
      } else if (a == "--threshold") {
        cfg.match_threshold = std::stof(requireArg(i, argc, argv, "--threshold"));
      } else if (a == "-h" || a == "--help") {
        printUsage(argv[0]);
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + a);
      }
    }

    auto engine = std::make_shared<facerec::Engine>(cfg);
    std::mutex engine_mu;

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
      res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Post("/v1/register", [&](const httplib::Request& req, httplib::Response& res) {
      try {
        const json body = json::parse(req.body);
        const std::string name = body.at("name").get<std::string>();
        const std::string b64 = body.at("image_base64").get<std::string>();
        const auto bytes = base64Decode(b64);
        const cv::Mat img = facerec::decodeImage(bytes);
        if (img.empty()) {
          throw std::runtime_error("could not decode image");
        }
        std::string id;
        {
          std::lock_guard<std::mutex> lock(engine_mu);
          id = engine->registerFace(img, name);
        }
        res.set_content(json{{"id", id}, {"name", name}}.dump(), "application/json");
      } catch (const std::exception& ex) {
        res.status = 400;
        res.set_content(json{{"error", ex.what()}}.dump(), "application/json");
      }
    });

    svr.Post("/v1/recognize", [&](const httplib::Request& req, httplib::Response& res) {
      try {
        const json body = json::parse(req.body);
        const std::string b64 = body.at("image_base64").get<std::string>();
        int top_k = 1;
        if (body.contains("top_k")) {
          top_k = body.at("top_k").get<int>();
        }
        const auto bytes = base64Decode(b64);
        const cv::Mat img = facerec::decodeImage(bytes);
        if (img.empty()) {
          throw std::runtime_error("could not decode image");
        }
        std::vector<facerec::Match> matches;
        {
          std::lock_guard<std::mutex> lock(engine_mu);
          matches = engine->recognize(img, top_k);
        }
        json out;
        out["matches"] = json::array();
        for (const auto& m : matches) {
          out["matches"].push_back({{"id", m.id}, {"name", m.name}, {"score", m.score}});
        }
        res.set_content(out.dump(), "application/json");
      } catch (const std::exception& ex) {
        res.status = 400;
        res.set_content(json{{"error", ex.what()}}.dump(), "application/json");
      }
    });

    svr.Get("/v1/faces", [&](const httplib::Request&, httplib::Response& res) {
      try {
        std::vector<std::pair<std::string, std::string>> people;
        {
          std::lock_guard<std::mutex> lock(engine_mu);
          people = engine->listPersons();
        }
        json out = json::array();
        for (const auto& p : people) {
          out.push_back({{"id", p.first}, {"name", p.second}});
        }
        res.set_content(out.dump(), "application/json");
      } catch (const std::exception& ex) {
        res.status = 500;
        res.set_content(json{{"error", ex.what()}}.dump(), "application/json");
      }
    });

    svr.Delete(R"(/v1/faces/(\w+))", [&](const httplib::Request& req, httplib::Response& res) {
      try {
        const std::string id = req.matches[1];
        bool ok = false;
        {
          std::lock_guard<std::mutex> lock(engine_mu);
          ok = engine->removePerson(id);
        }
        if (!ok) {
          res.status = 404;
          res.set_content(json{{"error", "not found"}}.dump(), "application/json");
          return;
        }
        res.set_content(json{{"deleted", id}}.dump(), "application/json");
      } catch (const std::exception& ex) {
        res.status = 400;
        res.set_content(json{{"error", ex.what()}}.dump(), "application/json");
      }
    });

    std::cout << "facerec_server listening on http://" << host << ":" << port << std::endl;
    if (!svr.listen(host, port)) {
      throw std::runtime_error("failed to bind " + host + ":" + std::to_string(port));
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
