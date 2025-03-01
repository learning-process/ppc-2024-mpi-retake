#include "seq/sharamygina_i_vector_dot_product/include/ops_seq.h"

bool sharamygina_i_vector_dot_product_seq::vector_dot_product_seq::PreProcessingImpl() {
  v1.resize(task_data->inputs_count[0]);
  v2.resize(task_data->inputs_count[1]);
  auto* tempPtr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(tempPtr, tempPtr + task_data->inputs_count[0], v1.begin());
  tempPtr = reinterpret_cast<int*>(task_data->inputs[1]);
  std::copy(tempPtr, tempPtr + task_data->inputs_count[1], v2.begin());
  res = 0;
  return true;
}

bool sharamygina_i_vector_dot_product_seq::vector_dot_product_seq::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs.size() == task_data->inputs_count.size() && task_data->inputs.size() == 2) &&
         task_data->outputs_count[0] == 1 && (task_data->outputs.size() == task_data->outputs_count.size()) &&
         task_data->outputs.size() == 1;
}

bool sharamygina_i_vector_dot_product_seq::vector_dot_product_seq::RunImpl() {
  for (unsigned int i = 0; i < v1.size(); i++) {
    res += v1[i] * v2[i];
  }
  return true;
}

bool sharamygina_i_vector_dot_product_seq::vector_dot_product_seq::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = res;
  return true;
}
