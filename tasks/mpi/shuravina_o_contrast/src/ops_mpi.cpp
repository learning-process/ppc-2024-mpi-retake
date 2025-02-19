#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

bool shuravina_o_contrast::ContrastTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    const unsigned int input_size = task_data->inputs_count[0];
    auto *in_ptr = reinterpret_cast<uint8_t *>(task_data->inputs[0]);
    if (in_ptr == nullptr) {
      throw std::runtime_error("Input pointer is null");
    }
    input_ = std::vector<uint8_t>(in_ptr, in_ptr + input_size);

    const unsigned int output_size = task_data->outputs_count[0];
    output_ = std::vector<uint8_t>(output_size, 0);

    width_ = height_ = static_cast<int>(std::sqrt(input_size));
  }
  return true;
}

bool shuravina_o_contrast::ContrastTaskMPI::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

void shuravina_o_contrast::ContrastTaskMPI::IncreaseContrast() {
  const uint8_t min_val = *std::ranges::min_element(input_);
  const uint8_t max_val = *std::ranges::max_element(input_);

  if (min_val == max_val) {
    std::ranges::fill(output_, 255);
    return;
  }

  for (size_t i = 0; i < input_.size(); ++i) {
    output_[i] = static_cast<uint8_t>((input_[i] - min_val) * 255 / (max_val - min_val));
  }
}

bool shuravina_o_contrast::ContrastTaskMPI::RunImpl() {
  if (world_.rank() == 0) {
    IncreaseContrast();
  }

  MPI_Bcast(output_.data(), static_cast<int>(output_.size()), MPI_BYTE, 0, MPI_COMM_WORLD);

  return true;
}

bool shuravina_o_contrast::ContrastTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(output_, reinterpret_cast<uint8_t *>(task_data->outputs[0]));
  }
  return true;
}