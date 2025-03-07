#include "seq/sharamygina_i_vector_dot_product/include/ops_seq.h"

#include <algorithm>

bool sharamygina_i_vector_dot_product_seq::VectorDotProductSeq::PreProcessingImpl() {
  v1_.resize(task_data->inputs_count[0]);
  v2_.resize(task_data->inputs_count[1]);
  auto* temp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(temp_ptr, temp_ptr + task_data->inputs_count[0], v1_.begin());
  temp_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  std::copy(temp_ptr, temp_ptr + task_data->inputs_count[1], v2_.begin());
  res_ = 0;
  return true;
}

bool sharamygina_i_vector_dot_product_seq::VectorDotProductSeq::ValidationImpl() {
  return (task_data->inputs.size() == task_data->inputs_count.size() && task_data->inputs.size() == 2) &&
         (task_data->inputs_count[0] == task_data->inputs_count[1]) && task_data->outputs_count[0] == 1 &&
         (task_data->outputs.size() == task_data->outputs_count.size()) && task_data->outputs.size() == 1;
}

bool sharamygina_i_vector_dot_product_seq::VectorDotProductSeq::RunImpl() {
  for (unsigned int i = 0; i < v1_.size(); i++) {
    res_ += v1_[i] * v2_[i];
  }
  return true;
}

bool sharamygina_i_vector_dot_product_seq::VectorDotProductSeq::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = res_;
  return true;
}
