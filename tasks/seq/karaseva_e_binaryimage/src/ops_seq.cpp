#include "seq/karaseva_e_binaryimage/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

// LabelUnionFind class defined at the top to avoid errors
class LabelUnionFind {
 public:
  int Find(int label) {
    if (parent_.find(label) == parent_.end()) {
      return label;
    }
    if (parent_[label] != label) {
      parent_[label] = Find(parent_[label]);
    }
    return parent_[label];
  }

  void Unite(int label1, int label2) {
    int root1 = Find(label1);
    int root2 = Find(label2);
    if (root1 != root2) {
      parent_[root2] = root1;
    }
  }

 private:
  std::unordered_map<int, int> parent_;
};

namespace {
void FixLabels(std::vector<int>& labeled_image, int rows, int cols) {
  std::unordered_map<int, int> label_map;
  int next_label = 2;

  for (int i = 0; i < rows * cols; ++i) {
    if (labeled_image[i] > 1) {
      if (label_map.find(labeled_image[i]) == label_map.end()) {
        label_map[labeled_image[i]] = next_label++;
      }
      labeled_image[i] = label_map[labeled_image[i]];
    }
  }
}

bool IsValidNeighbor(int nx, int ny, int rows, int columns, const std::vector<int>& labeled_image) {
  return nx >= 0 && ny >= 0 && nx < rows && ny < columns && labeled_image[(nx * columns) + ny] > 1;
}

void ProcessNeighbors(int x, int y, int rows, int columns, const std::vector<int>& labeled_image,
                      std::vector<int>& neighbors, const int dx[], const int dy[]) {
  for (int i = 0; i < 3; ++i) {
    int nx = x + dx[i];
    int ny = y + dy[i];
    if (IsValidNeighbor(nx, ny, rows, columns, labeled_image)) {
      neighbors.push_back(labeled_image[(nx * columns) + ny]);
    }
  }
}

void AssignLabel(int position, int& current_label, std::vector<int>& labeled_image, const std::vector<int>& neighbors,
                 LabelUnionFind& label_union) {
  if (neighbors.empty()) {
    labeled_image[position] = current_label++;
  } else {
    int min_label = *std::ranges::min_element(neighbors);
    labeled_image[position] = min_label;

    for (int label : neighbors) {
      label_union.Unite(min_label, label);
    }
  }
}
}  // namespace

bool karaseva_e_binaryimage_seq::TestTaskSequential::PreProcessingImpl() {
  rows_ = static_cast<int>(task_data->inputs_count[0]);
  columns_ = static_cast<int>(task_data->inputs_count[1]);
  int pixels_count = rows_ * columns_;

  image_ = std::vector<int>(pixels_count);
  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + pixels_count, image_.begin());

  labeled_image_.assign(pixels_count, 1);  // The background remains 1, objects are marked with 2
  return true;
}

bool karaseva_e_binaryimage_seq::TestTaskSequential::ValidationImpl() {
  int tmp_rows = static_cast<int>(task_data->inputs_count[0]);
  int tmp_columns = static_cast<int>(task_data->inputs_count[1]);
  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);

  if (tmp_rows <= 0 || tmp_columns <= 0 || static_cast<int>(task_data->outputs_count[0]) != tmp_rows ||
      static_cast<int>(task_data->outputs_count[1]) != tmp_columns) {
    return false;
  }

  return std::all_of(tmp_ptr, tmp_ptr + (static_cast<std::size_t>(tmp_rows) * static_cast<std::size_t>(tmp_columns)),
                     [](int pixel) { return pixel == 0 || pixel == 1; });
}

bool karaseva_e_binaryimage_seq::TestTaskSequential::RunImpl() {
  int current_label = 2;
  LabelUnionFind label_union;
  const int dx[] = {-1, 0, -1};
  const int dy[] = {0, -1, 1};

  for (int x = 0; x < rows_; ++x) {
    for (int y = 0; y < columns_; ++y) {
      int position = (x * columns_) + y;
      if (image_[position] == 0) {
        std::vector<int> neighbors;
        ProcessNeighbors(x, y, rows_, columns_, labeled_image_, neighbors, dx, dy);
        AssignLabel(position, current_label, labeled_image_, neighbors, label_union);
      }
    }
  }

  for (int i = 0; i < rows_ * columns_; ++i) {
    if (labeled_image_[i] > 1) {
      labeled_image_[i] = label_union.Find(labeled_image_[i]);
    }
  }

  FixLabels(labeled_image_, rows_, columns_);
  return true;
}

bool karaseva_e_binaryimage_seq::TestTaskSequential::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(labeled_image_, output_ptr);
  return true;
}