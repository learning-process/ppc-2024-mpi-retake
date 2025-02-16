#include "seq/karaseva_e_reduce/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <numeric> 
#include <iostream>  
#include <vector>

bool karaseva_e_reduce_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);  // int

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);  

  rc_size_ = static_cast<int>(std::sqrt(input_size));

  return true;
}

bool karaseva_e_reduce_seq::TestTaskSequential::ValidationImpl() {
  // Check that the number of elements in the output is 1
  if (task_data->outputs_count[0] != 1) {
    return false;
  }

  // Check that the number of input data corresponds to the size of the matrix
  size_t expected_input_size = static_cast<size_t>(std::sqrt(task_data->inputs_count[0]));
  if (task_data->inputs_count[0] != expected_input_size * expected_input_size) {
    return false;
  }

  return true;
}

bool karaseva_e_reduce_seq::TestTaskSequential::RunImpl() {
  int sum = 0;
  for (const auto &val : input_) {
    sum += val;
  }

  // result of the reduce in output_
  output_[0] = sum;
  return true;
}

bool karaseva_e_reduce_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = output_[0];
  return true;
}