#include "seq/khovansky_d_num_of_alternations_signs/include/ops_seq.hpp"

bool khovansky_d_num_of_alternations_signs_seq::NumOfAlternationsSignsSeq::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input = std::vector<int>(in_ptr, in_ptr + input_size);

  res = 0;

  return true;
}

bool khovansky_d_num_of_alternations_signs_seq::NumOfAlternationsSignsSeq::ValidationImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
    return false;
  }

  if (task_data->outputs[0] == nullptr) {
    return false;
  }

  return task_data->inputs_count[0] >= 2 && task_data->outputs_count[0] == 1;
}

bool khovansky_d_num_of_alternations_signs_seq::NumOfAlternationsSignsSeq::RunImpl() {
  int input_size = input.size(); 
  for (int i = 0; i < input_size - 1; i++) {
    if ((input[i] < 0 && input[i + 1] >= 0) || (input[i] >= 0 && input[i + 1] < 0)) {
      res++;
    }
  }
  return true;
}

bool khovansky_d_num_of_alternations_signs_seq::NumOfAlternationsSignsSeq::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = res;

  return true;
}
