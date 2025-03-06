#include "seq/Shpynov_N_radix_sort/include/Shpynov_N_radix_sort.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool shpynov_n_radix_sort_seq::TestTaskSEQ::ValidationImpl() {
  if (task_data->inputs_count[0] == 0) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool shpynov_n_radix_sort_seq::TestTaskSEQ::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  result_ = std::vector<int>(output_size, 0);

  return true;
}

bool shpynov_n_radix_sort_seq::TestTaskSEQ::RunImpl() {
  result_ = RadixSort(input_);

  return true;
}

bool shpynov_n_radix_sort_seq::TestTaskSEQ::PostProcessingImpl() {
  auto *output = reinterpret_cast<int *>(task_data->outputs[0]);
  std::ranges::copy(result_.begin(), result_.end(), output);
  return true;
}
