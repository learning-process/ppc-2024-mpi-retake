#include "seq/shuravina_o_coontrast/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

bool shuravina_o_contrast::ContrastTaskSequential::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<uint8_t *>(task_data->inputs[0]);
  if (in_ptr == nullptr) {
    throw std::runtime_error("Input pointer is null");
  }
  input_ = std::vector<uint8_t>(in_ptr, in_ptr + task_data->inputs_count[0]);

  output_ = std::vector<uint8_t>(task_data->outputs_count[0], 0);

  const int size = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  width_ = height_ = size;
  return true;
}

bool shuravina_o_contrast::ContrastTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

void shuravina_o_contrast::ContrastTaskSequential::IncreaseContrast() {
  const uint8_t min_val = *std::ranges::min_element(input_);
  const uint8_t max_val = *std::ranges::max_element(input_);

  if (min_val == max_val) {
    std::ranges::copy(input_.begin(), input_.end(), output_.begin());
    return;
  }

  for (size_t i = 0; i < input_.size(); ++i) {
    output_[i] = static_cast<uint8_t>((input_[i] - min_val) * 255 / (max_val - min_val));
  }
}

bool shuravina_o_contrast::ContrastTaskSequential::RunImpl() {
  IncreaseContrast();
  return true;
}

bool shuravina_o_contrast::ContrastTaskSequential::PostProcessingImpl() {
  std::ranges::copy(output_.begin(), output_.end(), reinterpret_cast<uint8_t *>(task_data->outputs[0]));
  return true;
}