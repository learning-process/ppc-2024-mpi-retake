#include "seq/konkov_i_linear_hist_stretch_seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

bool konkov_i_linear_hist_stretch_seq::LinearHistStretchSeq::PreProcessingImpl() {
  size_t input_size = task_data->inputs_count[0];
  auto* input_ptr = reinterpret_cast<uint8_t*>(task_data->inputs[0]);
  input_ = std::vector<uint8_t>(input_ptr, input_ptr + input_size);

  size_t output_size = task_data->outputs_count[0];
  output_ = std::vector<uint8_t>(output_size, 0);

  auto [min_it, max_it] = std::ranges::minmax_element(input_);
  min_val_ = *min_it;
  max_val_ = *max_it;

  return true;
}

bool konkov_i_linear_hist_stretch_seq::LinearHistStretchSeq::ValidationImpl() {
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }
  if (task_data->inputs_count[0] == 0) {
    return false;
  }
  if (task_data->inputs[0] == nullptr) {
    return false;
  }
  return true;
}

bool konkov_i_linear_hist_stretch_seq::LinearHistStretchSeq::RunImpl() {
  if (min_val_ == max_val_) {
    output_ = input_;
    return true;
  }
  for (size_t i = 0; i < input_.size(); ++i) {
    output_[i] = static_cast<uint8_t>(255 * (input_[i] - min_val_) / (max_val_ - min_val_));
  }
  return true;
}

bool konkov_i_linear_hist_stretch_seq::LinearHistStretchSeq::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<uint8_t*>(task_data->outputs[0]));
  return true;
}