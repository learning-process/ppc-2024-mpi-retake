#include "mpi/konkov_i_linear_hist_stretch_mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

bool konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::PreProcessingImpl() {
  size_t input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<uint8_t*>(task_data->inputs[0]);
  input_ = std::vector<uint8_t>(in_ptr, in_ptr + input_size);
  output_.resize(input_size);
  return true;
}

bool konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::ValidationImpl() {
  boost::mpi::communicator comm;

  if (task_data->inputs_count.size() < 1 || task_data->outputs_count.size() < 1) {
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

void konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::ComputeLocalMinMax(uint8_t& local_min,
                                                                                uint8_t& local_max) {
  local_min = *std::min_element(input_.begin(), input_.end());
  local_max = *std::max_element(input_.begin(), input_.end());
}

bool konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::RunImpl() {
  uint8_t local_min, local_max;
  ComputeLocalMinMax(local_min, local_max);

  boost::mpi::all_reduce(world_, local_min, min_intensity_, boost::mpi::minimum<uint8_t>());
  boost::mpi::all_reduce(world_, local_max, max_intensity_, boost::mpi::maximum<uint8_t>());

  if (min_intensity_ == max_intensity_) {
    std::fill(output_.begin(), output_.end(), min_intensity_);
    return true;
  }

  ApplyLinearStretch();
  return true;
}

void konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::ApplyLinearStretch() {
  for (size_t i = 0; i < input_.size(); ++i) {
    output_[i] = static_cast<uint8_t>((input_[i] - min_intensity_) * 255.0 /
                                      (max_intensity_ - min_intensity_));
  }
}

bool konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI::PostProcessingImpl() {
  std::copy(output_.begin(), output_.end(), reinterpret_cast<uint8_t*>(task_data->outputs[0]));
  return true;
}
