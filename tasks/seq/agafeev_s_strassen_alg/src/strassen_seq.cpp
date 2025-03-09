#include "seq/agafeev_s_strassen_alg/include/strassen_seq.hpp"

namespace agafeev_s_strassen_alg_seq {

bool MultiplMatrixSequental::PreProcessingImpl() {
  auto* temp_ptr = reinterpret_cast<T*>(task_data->inputs[0]);
  first_input_.insert(input_.begin(), temp_ptr, temp_ptr + task_data->inputs_count[0]);
  auto* temp_ptr = reinterpret_cast<T*>(task_data->inputs[1]);
  second_input_.insert(input_.begin(), temp_ptr, temp_ptr + task_data->inputs_count[0]);

  return true;
}

template <typename T>
bool MultiplMatrixSequental<T>::ValidationImpl() {
  bool isPowerOfTwo = task_data->outputs_count[0] && !(task_data->outputs_count[0] & (task_data->outputs_count[0] - 1));
  return (task_data->outputs_count[0] == task_data->inputs_count[0] && isPowerOfTwo);
}

template <typename T>
bool MultiplMatrixSequental<T>::RunImpl() {
  result_ = matrix_Multiply(first_input_, second_input_, size_);

  return true;
}

template <typename T>
bool MultiplMatrixSequental<T>::PostProcessingImpl() {
  reinterpret_cast<T*>(task_data->outputs[0]) = result_;

  return true;
}

template class MultiplMatrixSequental<double>;

}  // namespace agafeev_s_strassen_alg_seq
