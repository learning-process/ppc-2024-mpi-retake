#include "mpi/strakhov_a_fox_algorithm/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool strakhov_a_fox_algorithm_mpi::TestTaskMPI::PreProcessingImpl() {
  rc_size_ = task_data->inputs_count[0];

  auto* in_ptr1 = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* in_ptr2 = reinterpret_cast<double*>(task_data->inputs[1]);
  matrA_ = std::vector<double>(in_ptr1, in_ptr1 + (rc_size_ * rc_size_));
  matrB_ = std::vector<double>(in_ptr2, in_ptr2 + (rc_size_ * rc_size_));

  if (world_.rank() >= static_cast<int>(rc_size_)) {
    return true;
  }
  return true;
}

bool strakhov_a_fox_algorithm_mpi::TestTaskMPI::ValidationImpl() {
  // Check equality of counts elements
  return ((task_data->inputs_count[0] * task_data->inputs_count[0]) == task_data->outputs_count[0]) &&
         (task_data->outputs_count[0] > 0);
}

bool strakhov_a_fox_algorithm_mpi::TestTaskMPI::RunImpl() {
  std::vector<double> output_local(rc_size_ * rc_size_, 0);
  size_t actual_size = world_.size();
  actual_size = std::min(actual_size, rc_size_);
  if (static_cast<int>(rc_size_) <= world_.rank()) {
    return true;
  }
  size_t d = rc_size_ / actual_size;
  size_t b = d * world_.rank();
  size_t e = b + d;
  if (world_.rank() == static_cast<int>(actual_size - 1)) {
    e = rc_size_;
  }
  for (int k = static_cast<int>(b); k < static_cast<int>(e); k++) {
    for (int i = 0; i < static_cast<int>(rc_size_); i++) {
      for (int j = 0; j < static_cast<int>(rc_size_); j++) {
        size_t x_a = (i + k + j) % rc_size_;
        size_t y_b = (i + j + k) % rc_size_;
        double ans = matrA_[x_a + (i * static_cast<int>(rc_size_))] * matrB_[((y_b * static_cast<int>(rc_size_)) + j)];
        output_local[(static_cast<int>(rc_size_) * i) + j] += ans;
      }
    }
  }
  if (world_.rank() == 0) {
    std::vector<double> local_ans(static_cast<int>(rc_size_) * static_cast<int>(rc_size_), 0);
    for (int i = 1; i < static_cast<int>(actual_size); i++) {
      world_.recv(i, 0, local_ans.data(), static_cast<int>(rc_size_) * static_cast<int>(rc_size_));
      for (size_t j = 0; j < local_ans.size(); j++) {
        output_local[j] += local_ans[j];
      }
    }
    output_ = output_local;
  } else {
    world_.send(0, 0, output_local.data(), static_cast<int>(output_local.size()));
  }
  return true;
}

bool strakhov_a_fox_algorithm_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() != 0) {
    return true;
  }
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
