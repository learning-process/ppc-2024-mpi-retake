#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool shuravina_o_contrast::TestTaskMPI::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<uint8_t *>(task_data->inputs[0]);
  input_ = std::vector<uint8_t>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<uint8_t>(output_size, 0);

  rc_size_ = static_cast<int>(std::sqrt(input_size));
  return true;
}

bool shuravina_o_contrast::TestTaskMPI::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

void shuravina_o_contrast::TestTaskMPI::IncreaseContrast() {
  uint8_t min_val = *std::min_element(input_.begin(), input_.end());
  uint8_t max_val = *std::max_element(input_.begin(), input_.end());

  for (size_t i = 0; i < input_.size(); ++i) {
    output_[i] = static_cast<uint8_t>((input_[i] - min_val) * 255 / (max_val - min_val));
  }
}

bool shuravina_o_contrast::TestTaskMPI::RunImpl() {
  if (world_.rank() == 0) {
    IncreaseContrast();
  }
  world_.barrier();
  return true;
}

bool shuravina_o_contrast::TestTaskMPI::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<uint8_t *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}