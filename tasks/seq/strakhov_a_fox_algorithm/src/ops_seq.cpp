#include "seq/strakhov_a_fox_algorithm/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

bool strakhov_a_fox_algorithm::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  rc_size_ = task_data->inputs_count[0];
  auto *in_ptr1 = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *in_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
  matrA_ = std::vector<double>(in_ptr1, in_ptr1 + rc_size_ * rc_size_);
  matrB_ = std::vector<double>(in_ptr2, in_ptr2 + rc_size_ * rc_size_);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0);
  return true;
}

bool strakhov_a_fox_algorithm::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  return ((task_data->inputs_count[0] * task_data->inputs_count[0]) == task_data->outputs_count[0]) &&
         (task_data->outputs_count[0] > 0);
}

bool strakhov_a_fox_algorithm::TestTaskSequential::RunImpl() {
  for (int k = 0; k < rc_size_; k++) {
    for (int i = 0; i < rc_size_; i++) {
      for (int j = 0; j < rc_size_; j++) {
        size_t xA = (i + k + j) % rc_size_;
        size_t yB = (i + j + k) % rc_size_;
        double ans = matrA_[xA + (i * rc_size_)] * matrB_[(yB * rc_size_ + j)];
        output_[(rc_size_ * i) + j] += ans;  // first hag po x +  first hag po y + sdig ,
      }
    }
  }
  return true;
}

bool strakhov_a_fox_algorithm::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
