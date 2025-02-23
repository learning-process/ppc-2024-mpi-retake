#include "seq/karaseva_e_binaryimage/include/ops_seq.hpp"

#include <algorithm>
#include <ranges>
#include <unordered_map>
#include <vector>

// Function moved to anonymous namespace as per the error message
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
}  // namespace

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

  return std::all_of(tmp_ptr, tmp_ptr + static_cast<size_t>(tmp_rows) * tmp_columns,
                     [](int pixel) { return pixel == 0 || pixel == 1; });
}

bool karaseva_e_binaryimage_seq::TestTaskSequential::RunImpl() {
  int current_label = 2;
  LabelUnionFind label_union;
  int dx[] = {-1, 0, -1};
  int dy[] = {0, -1, 1};

  for (int x = 0; x < rows_; ++x) {
    for (int y = 0; y < columns_; ++y) {
      int position = x * columns_ + y;
      if (image_[position] == 0) {
        std::vector<int> neighbors;

        for (int i = 0; i < 3; ++i) {
          int nx = x + dx[i];
          int ny = y + dy[i];
          int neighbor_pos = nx * columns_ + ny;

          if (nx >= 0 && ny >= 0 && nx < rows_ && ny < columns_ && labeled_image_[neighbor_pos] > 1) {
            neighbors.push_back(labeled_image_[neighbor_pos]);
          }
        }

        if (neighbors.empty()) {
          labeled_image_[position] = current_label++;
        } else {
          int min_label = *std::ranges::min_element(neighbors);
          labeled_image_[position] = min_label;

          for (int label : neighbors) {
            label_union.Unite(min_label, label);
          }
        }
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