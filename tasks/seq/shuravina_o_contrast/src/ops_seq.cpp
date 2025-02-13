#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool shuravina_o_contrast::ContrastTaskSequential::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<uint8_t *>(task_data->inputs[0]);
  input_ = std::vector<uint8_t>(in_ptr, in_ptr + task_data->inputs_count[0]);

  output_ = std::vector<uint8_t>(task_data->outputs_count[0], 0);

  width_ = height_ = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  return true;
}

bool shuravina_o_contrast::ContrastTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool shuravina_o_contrast::ContrastTaskSequential::RunImpl() {
  uint8_t min_intensity = *std::min_element(input_.begin(), input_.end());
  uint8_t max_intensity = *std::max_element(input_.begin(), input_.end());

  for (size_t i = 0; i < input_.size(); ++i) {
    output_[i] = static_cast<uint8_t>((input_[i] - min_intensity) * 255 / (max_intensity - min_intensity));
  }
  return true;
}

bool shuravina_o_contrast::ContrastTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<uint8_t *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}