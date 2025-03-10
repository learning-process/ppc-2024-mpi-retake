#include "mpi/Konstantinov_I_histogram_stretching/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <vector>

bool konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchSeq::PreProcessingImpl() {
  int size = static_cast<int>(task_data->inputs_count[0]);
  image_input_ = std::vector<int>(size);
  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + size, image_input_.begin());

  int pixel_count = size / 3;
  I_.resize(pixel_count);
  for (int i = 0, k = 0; i < size; i += 3, ++k) {
    int r = image_input_[i];
    int g = image_input_[i + 1];
    int b = image_input_[i + 2];

    I_[k] = static_cast<int>((0.299 * static_cast<double>(r)) + (0.587 * static_cast<double>(g)) +
                             (0.114 * static_cast<double>(b)));
  }

  image_output_ = {};
  return true;
}

bool konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchSeq::ValidationImpl() {
  int size = static_cast<int>(task_data->inputs_count[0]);
  if (size % 3 != 0) {
    return false;
  }
  for (int i = 0; i < size; ++i) {
    int value = reinterpret_cast<int*>(task_data->inputs[0])[i];
    if (value < 0 || value > 255) {
      return false;
    }
  }

  return ((!task_data->inputs.empty() && !task_data->outputs.empty()) &&
          (!task_data->inputs_count.empty() && task_data->inputs_count[0] != 0) &&
          (!task_data->outputs_count.empty() && task_data->outputs_count[0] != 0));
}

bool konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchSeq::RunImpl() {
  int size = static_cast<int>(image_input_.size());
  image_output_.resize(size);
  int imin = 255;
  int imax = 0;

  for (int intensity : I_) {
    imin = std::min(imin, intensity);
    imax = std::max(imax, intensity);
  }

  if (imin == imax) {
    image_output_ = image_input_;
    return true;
  }

  for (int i = 0, k = 0; i < size; i += 3, ++k) {
    int inew = ((I_[k] - imin) * 255) / (imax - imin);

    float coeff = static_cast<float>(inew) / static_cast<float>(I_[k]);

    image_output_[i] = std::min(255, static_cast<int>(static_cast<float>(image_input_[i]) * coeff));
    image_output_[i + 1] = std::min(255, static_cast<int>(static_cast<float>(image_input_[i + 1]) * coeff));
    image_output_[i + 2] = std::min(255, static_cast<int>(static_cast<float>(image_input_[i + 2]) * coeff));
  }

  return true;
}

bool konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchSeq::PostProcessingImpl() {
  auto* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(image_output_, output);
  return true;
}

bool konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    int size = static_cast<int>(task_data->inputs_count[0]);
    image_input_ = std::vector<int>(size);
    auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + size, image_input_.begin());
    image_output_ = {};
    return true;
  }
  return true;
}

bool konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    int size = static_cast<int>(task_data->inputs_count[0]);
    if (size % 3 != 0) {
      return false;
    }
    for (int i = 0; i < size; ++i) {
      int value = reinterpret_cast<int*>(task_data->inputs[0])[i];
      if (value < 0 || value > 255) {
        return false;
      }
    }

    return ((!task_data->inputs.empty() && !task_data->outputs.empty()) &&
            (!task_data->inputs_count.empty() && task_data->inputs_count[0] != 0) &&
            (!task_data->outputs_count.empty() && task_data->outputs_count[0] != 0));
  }
  return true;
}

bool konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi::RunImpl() {
  int size = 0;
  if (world_.rank() == 0) {
    size = static_cast<int>(image_input_.size());
  }

  boost::mpi::broadcast(world_, size, 0);

  int num_pixels = 0;
  int pixels_per_process = 0;
  int extra_pixels = 0;
  if (world_.rank() == 0) {
    num_pixels = size / 3;
    pixels_per_process = num_pixels / world_.size();
    extra_pixels = num_pixels % world_.size();
  }

  boost::mpi::broadcast(world_, num_pixels, 0);
  boost::mpi::broadcast(world_, pixels_per_process, 0);
  boost::mpi::broadcast(world_, extra_pixels, 0);

  int local_pixels = pixels_per_process + (world_.rank() < extra_pixels ? 1 : 0);

  std::vector<int> offset(world_.size(), 0);
  std::vector<int> send_counts(world_.size(), 0);

  if (world_.rank() == 0) {
    for (int proc = 0; proc < world_.size(); ++proc) {
      send_counts[proc] = (pixels_per_process + (proc < extra_pixels ? 1 : 0)) * 3;
      if (proc > 0) {
        offset[proc] = offset[proc - 1] + send_counts[proc - 1];
      }
    }
  }

  boost::mpi::broadcast(world_, send_counts.data(), static_cast<int>(send_counts.size()), 0);
  boost::mpi::broadcast(world_, offset.data(), static_cast<int>(offset.size()), 0);

  std::vector<int> local_input(local_pixels * 3);
  boost::mpi::scatterv(world_, image_input_.data(), send_counts, offset, local_input.data(), local_pixels * 3, 0);

  int local_imin = 255;
  int local_imax = 0;
  std::vector<int> local_i(local_pixels);
  for (int i = 0, k = 0; i < local_pixels * 3; i += 3, ++k) {
    int r = local_input[i];
    int g = local_input[i + 1];
    int b = local_input[i + 2];

    local_i[k] = static_cast<int>((0.299 * static_cast<double>(r)) + (0.587 * static_cast<double>(g)) +
                                  (0.114 * static_cast<double>(b)));
    local_imin = std::min(local_imin, local_i[k]);
    local_imax = std::max(local_imax, local_i[k]);
  }

  int global_imin = 0;
  int global_imax = 0;
  boost::mpi::all_reduce(world_, local_imin, global_imin, boost::mpi::minimum<int>());
  boost::mpi::all_reduce(world_, local_imax, global_imax, boost::mpi::maximum<int>());

  if (global_imin == global_imax) {
    if (world_.rank() == 0) {
      image_output_ = image_input_;
    }
    return true;
  }

  std::vector<int> local_output(local_pixels * 3);
  for (int i = 0, k = 0; i < local_pixels * 3; i += 3, ++k) {
    int inew = ((local_i[k] - global_imin) * 255) / (global_imax - global_imin);
    float coeff = static_cast<float>(inew) / static_cast<float>(local_i[k]);

    local_output[i] = std::min(255, static_cast<int>(static_cast<float>(local_input[i]) * coeff));
    local_output[i + 1] = std::min(255, static_cast<int>(static_cast<float>(local_input[i + 1]) * coeff));
    local_output[i + 2] = std::min(255, static_cast<int>(static_cast<float>(local_input[i + 2]) * coeff));
  }

  if (world_.rank() == 0) {
    image_output_.resize(size);
  }

  boost::mpi::gatherv(world_, local_output.data(), local_pixels * 3, image_output_.data(), send_counts, offset, 0);

  return true;
}

bool konstantinov_i_linear_histogram_stretch_mpi::LinearHistogramStretchMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output = reinterpret_cast<int*>(task_data->outputs[0]);
    std::ranges::copy(image_output_, output);
  }
  return true;
}