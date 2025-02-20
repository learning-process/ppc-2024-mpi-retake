#include "seq/malyshev_v_lent_horizontal/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

bool malyshev_v_matrix_vector_seq::MatrixVectorMultiplication::PreProcessingImpl() {
  auto* matrix_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* vector_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  rows_ = task_data->inputs_count[0];
  cols_ = task_data->inputs_count[1];

  matrix_.assign(matrix_ptr, matrix_ptr + rows_ * cols_);
  vector_.assign(vector_ptr, vector_ptr + cols_);
  result_.resize(rows_, 0.0);

  return true;
}

bool malyshev_v_matrix_vector_seq::MatrixVectorMultiplication::ValidationImpl() {
  return task_data->inputs_count[1] == task_data->inputs_count[2];
}

bool malyshev_v_matrix_vector_seq::MatrixVectorMultiplication::RunImpl() {
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      result_[i] += matrix_[i * cols_ + j] * vector_[j];
    }
  }
  return true;
}

bool malyshev_v_matrix_vector_seq::MatrixVectorMultiplication::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(result_.begin(), result_.end(), output_ptr);
  return true;
}