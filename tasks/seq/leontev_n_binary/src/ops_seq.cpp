#include "seq/leontev_n_binary/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leontev_n_binary_seq {

bool CompNotZero(uint32_t a, uint32_t b) {
  if (a == 0) {
    return false;
  }
  if (b == 0) {
    return true;
  }
  return a < b;
}

size_t BinarySegmentsSeq::GetIndex(size_t i, size_t j) { return i * cols_ + j; }

bool BinarySegmentsSeq::ValidationImpl() {
  return !task_data->inputs.empty() && !task_data->outputs.empty() && task_data->inputs_count.size() == 2 &&
         task_data->outputs_count.size() == 2 && task_data->inputs_count[0] == task_data->outputs_count[0] &&
         task_data->inputs_count[1] == task_data->outputs_count[1];
}

bool BinarySegmentsSeq::PreProcessingImpl() {
  rows_ = task_data->inputs_count[0];
  cols_ = task_data->inputs_count[1];
  size_t total_pixels = rows_ * cols_;
  input_image_.resize(total_pixels);
  std::copy_n(reinterpret_cast<uint8_t*>(task_data->inputs[0]), total_pixels, input_image_.begin());
  return true;
}

bool BinarySegmentsSeq::RunImpl() {
  std::unordered_map<uint32_t, uint32_t> label_equivalences;
  uint32_t cur_label = 1;
  labels_.resize(rows_ * cols_, 0);
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      if (input_image_[GetIndex(row, col)] == 0) {
        continue;
      }

      uint32_t label_B = (col > 0) ? labels_[GetIndex(row, col - 1)] : 0;
      uint32_t label_C = (row > 0) ? labels_[GetIndex(row - 1, col)] : 0;
      uint32_t label_D = (row > 0 && col > 0) ? labels_[GetIndex(row - 1, col - 1)] : 0;

      if (label_B == 0 && label_C == 0 && label_D == 0) {
        labels_[GetIndex(row, col)] = cur_label++;
      } else {
        uint32_t min_label = std::min({label_B, label_C, label_D}, CompNotZero);
        labels_[GetIndex(row, col)] = min_label;
        for (uint32_t label : {label_B, label_C, label_D}) {
          if (label != 0 && label != min_label) {
            if (label != 0 && label != min_label && label > min_label) {
              label_equivalences[label] = min_label;
            }
            if (label != 0 && label != min_label && label < min_label) {
              label_equivalences[min_label] = label;
            }
          }
        }
      }
    }
  }
  for (auto& label : labels_) {
    while (label_equivalences.contains(label)) {
      label = label_equivalences[label];
    }
  }
  return true;
}

bool BinarySegmentsSeq::PostProcessingImpl() {
  size_t total_pixels = rows_ * cols_;
  std::copy_n(labels_.data(), total_pixels, reinterpret_cast<uint32_t*>(task_data->outputs[0]));
  return true;
}

}  // namespace leontev_n_binary_seq