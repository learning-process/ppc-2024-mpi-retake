#include "mpi/konkov_i_linear_hist_stretch_mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/all_reduce.hpp"
#include "boost/mpi/communicator.hpp"
#include "boost/mpi/operations.hpp"

bool konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::PreProcessingImpl() {
  size_t input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<uint8_t*>(task_data->inputs[0]);
  input_ = std::vector<uint8_t>(in_ptr, in_ptr + input_size);
  output_.resize(input_size);
  return true;
}

bool konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::ValidationImpl() {
  boost::mpi::communicator comm;

  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }

  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }

  if (task_data->inputs_count[0] == 0) {
    return false;
  }

  if (comm.rank() == 0) {
    if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
      return false;
    }
  }

  return true;
}

std::pair<uint8_t, uint8_t> konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::ComputeLocalMinMax() {
  if (input_.empty()) {
    return {0, 0};
  }
  uint8_t min_value = *std::ranges::min_element(input_);
  uint8_t max_value = *std::ranges::max_element(input_);
  return {min_value, max_value};
}

bool konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::RunImpl() {
  auto [local_min, local_max] = ComputeLocalMinMax();

  boost::mpi::all_reduce(world_, local_min, min_intensity_, boost::mpi::minimum<uint8_t>());
  boost::mpi::all_reduce(world_, local_max, max_intensity_, boost::mpi::maximum<uint8_t>());

  if (min_intensity_ == max_intensity_) {
    std::ranges::fill(output_, min_intensity_);
    return true;
  }

  ApplyLinearStretch();
  return true;
}

void konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::ApplyLinearStretch() {
  for (size_t i = 0; i < input_.size(); ++i) {
    output_[i] = static_cast<uint8_t>((input_[i] - min_intensity_) * 255.0 / (max_intensity_ - min_intensity_));
  }
}

bool konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<uint8_t*>(task_data->outputs[0]));
  return true;
}
