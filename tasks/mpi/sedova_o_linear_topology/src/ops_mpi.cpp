#include "mpi/sedova_o_linear_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <cmath>
#include <cstddef>
#include <vector>

bool sedova_o_linear_topology_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.size() == 1) {
    return true;
  }
  unsigned int input_size = 0;
  if (world_.rank() == 0) {
    output_.resize(0);
    input_size = task_data->inputs_count[0];
    int *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    input_.resize(input_size);
    std::copy(tmp_ptr, tmp_ptr + input_size, input_.begin());
  }

  return true;
}

bool sedova_o_linear_topology_mpi::TestTaskMPI::ValidationImpl() {
  return (world_.rank() != 0) || (task_data->inputs_count[0] > 0);
}

bool sedova_o_linear_topology_mpi::TestTaskMPI::RunImpl() {
  unsigned int input_size = 0;

  if (world_.rank() == 0) {
    // Multiply matrices
    for (int i = 0; i < rc_size_; ++i) {
      for (int j = 0; j < rc_size_; ++j) {
        for (int k = 0; k < rc_size_; ++k) {
          output_[(i * rc_size_) + j] += input_[(i * rc_size_) + k] * input_[(k * rc_size_) + j];
        }
      }
    }
  } else {
    // Multiply matrices
    for (int j = 0; j < rc_size_; ++j) {
      for (int k = 0; k < rc_size_; ++k) {
        for (int i = 0; i < rc_size_; ++i) {
          output_[(i * rc_size_) + j] += input_[(i * rc_size_) + k] * input_[(k * rc_size_) + j];
        }
      }
    }
  }
  world_.barrier();
  return true;
}

bool nesterov_a_test_task_mpi::TestTaskMPI::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
