#include "seq/leontev_n_binary/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <vector>

namespace leontev_n_binary_seq {

namespace {
bool CompNotZero(uint32_t a, uint32_t b) {
  if (a == 0) {
    return false;
  }
  if (b == 0) {
    return true;
  }
  return a < b;
}

void AppendEqs(std::vector<std::set<uint32_t>>& label_equivalences, uint32_t label1, uint32_t label2) {
  bool flag1 = false;
  bool flag2 = false;
  size_t l1id = 0;
  size_t l2id = 0;
  for (size_t i = 0; i < label_equivalences.size(); i++) {
    if (label_equivalences[i].contains(label1)) {
      flag1 = true;
      l1id = i;
    }
    if (label_equivalences[i].contains(label2)) {
      flag2 = true;
      l2id = i;
    }
  }
  if (flag1 && flag2) {
    if (l1id != l2id) {
      label_equivalences[l1id].merge(label_equivalences[l2id]);
      label_equivalences[l2id] = std::set<uint32_t>();
    }
  } else if (flag1 && !flag2) {
    label_equivalences[l1id].insert(label2);
  } else if (!flag1 && flag2) {
    label_equivalences[l2id].insert(label1);
  } else {
    label_equivalences.emplace_back(std::set<uint32_t>({label1, label2}));
  }
}
}  // namespace

size_t BinarySegmentsSeq::GetIndex(size_t i, size_t j) const { return (i * cols_) + j; }

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

void BinarySegmentsSeq::LoopProcess(size_t col, size_t row, uint32_t& cur_label,
                                    std::vector<std::set<uint32_t>>& label_equivalences) {
  if (input_image_[GetIndex(row, col)] == 0) {
    return;
  }
  uint32_t label_b = (col > 0) ? labels_[GetIndex(row, col - 1)] : 0;
  uint32_t label_c = (row > 0) ? labels_[GetIndex(row - 1, col)] : 0;
  uint32_t label_d = (row > 0 && col > 0) ? labels_[GetIndex(row - 1, col - 1)] : 0;

  if (label_b == 0 && label_c == 0 && label_d == 0) {
    labels_[GetIndex(row, col)] = cur_label++;
  } else {
    uint32_t min_label = std::min({label_b, label_c, label_d}, CompNotZero);
    labels_[GetIndex(row, col)] = min_label;
    for (uint32_t label : {label_b, label_c, label_d}) {
      if (label != 0 && label != min_label) {
        AppendEqs(label_equivalences, std::max(label, min_label), std::min(label, min_label));
      }
    }
  }
}

bool BinarySegmentsSeq::RunImpl() {
  std::vector<std::set<uint32_t>> label_equivalences;
  uint32_t cur_label = 1;
  labels_.resize(rows_ * cols_, 0);
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      LoopProcess(col, row, cur_label, label_equivalences);
    }
  }
  for (auto& label : labels_) {
    for (size_t i = 0; i < label_equivalences.size(); i++) {
      if (label_equivalences[i].contains(label)) {
        label = *std::min_element(label_equivalences[i].begin(), label_equivalences[i].end());
      }
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
