#include "gallery.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace facerec {
namespace {

std::string makeId() {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> dist;
  std::ostringstream oss;
  oss << std::hex << dist(rng) << dist(rng);
  return oss.str();
}

}  // namespace

Gallery::Gallery(std::string gallery_dir) : gallery_dir_(std::move(gallery_dir)) {
  fs::create_directories(fs::path(gallery_dir_) / "embeddings");
}

std::string Gallery::indexPath() const {
  return (fs::path(gallery_dir_) / "index.json").string();
}

std::string Gallery::embeddingsPath(const std::string& id) const {
  return (fs::path(gallery_dir_) / "embeddings" / (id + ".bin")).string();
}

void Gallery::writeEmbedding(const std::string& id, const std::vector<float>& emb) const {
  const auto path = embeddingsPath(id);
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to write embedding: " + path);
  }
  out.write(reinterpret_cast<const char*>(emb.data()),
            static_cast<std::streamsize>(emb.size() * sizeof(float)));
}

std::vector<float> Gallery::readEmbedding(const std::string& id) const {
  const auto path = embeddingsPath(id);
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("failed to read embedding: " + path);
  }
  const auto bytes = in.tellg();
  if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
    throw std::runtime_error("corrupt embedding: " + path);
  }
  in.seekg(0);
  std::vector<float> emb(static_cast<std::size_t>(bytes) / sizeof(float));
  in.read(reinterpret_cast<char*>(emb.data()), bytes);
  return emb;
}

void Gallery::rebuildMatrixLocked() {
  matrix_.clear();
  dim_ = 0;
  if (people_.empty()) {
    return;
  }
  dim_ = static_cast<int>(people_.front().embedding.size());
  matrix_.resize(people_.size() * static_cast<std::size_t>(dim_));
  for (std::size_t i = 0; i < people_.size(); ++i) {
    if (static_cast<int>(people_[i].embedding.size()) != dim_) {
      throw std::runtime_error("inconsistent embedding dimension in gallery");
    }
    std::copy(people_[i].embedding.begin(), people_[i].embedding.end(),
              matrix_.begin() + static_cast<std::ptrdiff_t>(i * dim_));
  }
}

void Gallery::load() {
  std::lock_guard<std::mutex> lock(mu_);
  people_.clear();
  matrix_.clear();
  dim_ = 0;

  const auto path = indexPath();
  if (!fs::exists(path)) {
    saveIndex();  // create empty index
    return;
  }

  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open gallery index: " + path);
  }
  json root;
  in >> root;
  if (!root.contains("persons") || !root["persons"].is_array()) {
    return;
  }

  for (const auto& p : root["persons"]) {
    PersonRecord rec;
    rec.id = p.at("id").get<std::string>();
    rec.name = p.at("name").get<std::string>();
    rec.embedding = readEmbedding(rec.id);
    people_.push_back(std::move(rec));
  }
  rebuildMatrixLocked();
}

void Gallery::saveIndex() const {
  json root;
  root["persons"] = json::array();
  for (const auto& p : people_) {
    root["persons"].push_back({{"id", p.id},
                               {"name", p.name},
                               {"embedding", "embeddings/" + p.id + ".bin"}});
  }
  const auto path = indexPath();
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write gallery index: " + path);
  }
  out << root.dump(2);
}

std::string Gallery::add(const std::string& name, const std::vector<float>& embedding) {
  if (name.empty()) {
    throw std::runtime_error("name must not be empty");
  }
  if (embedding.empty()) {
    throw std::runtime_error("embedding must not be empty");
  }

  std::lock_guard<std::mutex> lock(mu_);
  if (dim_ != 0 && static_cast<int>(embedding.size()) != dim_) {
    throw std::runtime_error("embedding dimension mismatch");
  }

  PersonRecord rec;
  rec.id = makeId();
  rec.name = name;
  rec.embedding = embedding;
  writeEmbedding(rec.id, rec.embedding);
  people_.push_back(std::move(rec));
  rebuildMatrixLocked();
  saveIndex();
  return people_.back().id;
}

bool Gallery::remove(const std::string& id) {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = std::find_if(people_.begin(), people_.end(),
                               [&](const PersonRecord& p) { return p.id == id; });
  if (it == people_.end()) {
    return false;
  }
  const auto emb_path = embeddingsPath(id);
  people_.erase(it);
  rebuildMatrixLocked();
  saveIndex();
  std::error_code ec;
  fs::remove(emb_path, ec);
  return true;
}

std::vector<std::pair<PersonRecord, float>> Gallery::match(
    const std::vector<float>& query, int top_k, float threshold) const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::pair<PersonRecord, float>> hits;
  if (people_.empty() || query.empty() || top_k <= 0) {
    return hits;
  }
  if (static_cast<int>(query.size()) != dim_) {
    throw std::runtime_error("query embedding dimension mismatch");
  }

  const std::size_t n = people_.size();
  std::vector<std::pair<float, std::size_t>> scores;
  scores.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    const float* row = matrix_.data() + i * static_cast<std::size_t>(dim_);
    float dot = 0.f;
    for (int d = 0; d < dim_; ++d) {
      dot += row[d] * query[static_cast<std::size_t>(d)];
    }
    if (dot >= threshold) {
      scores.emplace_back(dot, i);
    }
  }

  const auto limit = static_cast<std::size_t>(top_k);
  if (scores.size() > limit) {
    std::partial_sort(scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(limit),
                      scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    scores.resize(limit);
  } else {
    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
  }

  hits.reserve(scores.size());
  for (const auto& s : scores) {
    hits.emplace_back(people_[s.second], s.first);
  }
  return hits;
}

std::vector<std::pair<std::string, std::string>> Gallery::list() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::pair<std::string, std::string>> out;
  out.reserve(people_.size());
  for (const auto& p : people_) {
    out.emplace_back(p.id, p.name);
  }
  return out;
}

std::size_t Gallery::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return people_.size();
}

}  // namespace facerec
