// Copyright 2024 Nesterov Alexander
#include "seq/kalinin_d_vector_dot_product/include/ops_seq.hpp"

#include <cstddef>
#include <vector>
bool kalinin_d_vector_dot_product_seq::TestTaskSequential::ValidationImpl() {
  // Check count elements of output
  return (task_data->inputs.size() == task_data->inputs_count.size() && task_data->inputs.size() == 2) &&
         (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->outputs.size() == task_data->outputs_count.size()) && task_data->outputs.size() == 1 &&
         task_data->outputs_count[0] == 1;
}

bool kalinin_d_vector_dot_product_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output

  input_ = std::vector<std::vector<int>>(task_data->inputs.size());
  for (size_t i = 0; i < input_.size(); i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[i]);
    input_[i] = std::vector<int>(task_data->inputs_count[i]);
    for (size_t j = 0; j < task_data->inputs_count[i]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  res_ = 0;
  return true;
}

bool kalinin_d_vector_dot_product_seq::TestTaskSequential::RunImpl() {
  for (size_t i = 0; i < input_[0].size(); i++) {
    res_ += input_[0][i] * input_[1][i];
  }

  return true;
}

bool kalinin_d_vector_dot_product_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = res_;
  return true;
}

int kalinin_d_vector_dot_product_seq::VectorDotProduct(const std::vector<int>& v1, const std::vector<int>& v2) {
  long long result = 0;
  for (size_t i = 0; i < v1.size(); i++) {
    result += v1[i] * v2[i];
  }
  return static_cast<int>(result);
}