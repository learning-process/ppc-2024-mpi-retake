#include "mpi/konkov_i_linear_hist_stretch_mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ranges>
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

void konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::ComputeLocalMinMax(uint8_t& out_min,
                                                                                uint8_t& out_max) {
  if (input_.empty()) {
    return;
  }
  out_min = *std::ranges::min_element(input_);
  out_max = *std::ranges::max_element(input_);
}

bool konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::RunImpl() {
  uint8_t local_min = 0;
  uint8_t local_max = 0;
  ComputeLocalMinMax(local_min, local_max);

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
