#include "seq/karaseva_e_num_of_alternations_signs/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

bool karaseva_e_num_of_alternations_signs_seq::AlternatingSignsSequential::PreProcessingImpl() {
  // Initialize input vector
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);

  // Output: single integer result
  output_.resize(1, 0);
  return true;
}

bool karaseva_e_num_of_alternations_signs_seq::AlternatingSignsSequential::ValidationImpl() {
  // At least two elements are needed to check alternation
  return task_data->inputs_count[0] >= 2 && task_data->outputs_count[0] == 1;
}

bool karaseva_e_num_of_alternations_signs_seq::AlternatingSignsSequential::RunImpl() {
  alternations_count_ = 0;

  for (size_t i = 1; i < input_.size(); ++i) {
    if (input_[i] * input_[i - 1] < 0) {
      ++alternations_count_;
    }
  }

  output_[0] = alternations_count_;
  return true;
}

bool karaseva_e_num_of_alternations_signs_seq::AlternatingSignsSequential::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = output_[0];
  return true;
}