#pragma once

#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace facerec {

struct PersonRecord {
  std::string id;
  std::string name;
  std::vector<float> embedding;  // L2-normalized
};

class Gallery {
 public:
  explicit Gallery(std::string gallery_dir);

  void load();
  void saveIndex() const;

  std::string add(const std::string& name, const std::vector<float>& embedding);
  bool remove(const std::string& id);

  // Scores are cosine similarity (dot product of L2-normalized vectors).
  std::vector<std::pair<PersonRecord, float>> match(
      const std::vector<float>& query, int top_k, float threshold) const;

  std::vector<std::pair<std::string, std::string>> list() const;
  std::size_t size() const;

 private:
  void rebuildMatrixLocked();
  void writeEmbedding(const std::string& id, const std::vector<float>& emb) const;
  std::vector<float> readEmbedding(const std::string& id) const;
  std::string embeddingsPath(const std::string& id) const;
  std::string indexPath() const;

  std::string gallery_dir_;
  mutable std::mutex mu_;
  std::vector<PersonRecord> people_;
  // Contiguous N x D matrix for fast matching (row-major).
  std::vector<float> matrix_;
  int dim_ = 0;
};

}  // namespace facerec
