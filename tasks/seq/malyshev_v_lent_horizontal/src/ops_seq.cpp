#include "seq/malyshev_v_lent_horizontal/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace malyshev_v_lent_horizontal_seq {

    std::vector<double> GetRandomMatrix(size_t rows, size_t cols) {
  std::vector<double> matrix(rows * cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      matrix[i * cols + j] = static_cast<double>(rand()) / RAND_MAX * 100.0;
    }
  }
  return matrix;
}

std::vector<double> GetRandomVector(size_t size) {
  std::vector<double> vector(size);
  for (size_t i = 0; i < size; ++i) {
    vector[i] = static_cast<double>(rand()) / RAND_MAX * 100.0;
  }
  return vector;
}

bool malyshev_v_lent_horizontal_seq::MatrixVectorMultiplication::PreProcessingImpl() {
  auto* matrix_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* vector_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  rows_ = task_data->inputs_count[0];
  cols_ = task_data->inputs_count[1];

  matrix_.assign(matrix_ptr, matrix_ptr + rows_ * cols_);
  vector_.assign(vector_ptr, vector_ptr + cols_);
  result_.resize(rows_, 0.0);

  return true;
}

bool malyshev_v_lent_horizontal_seq::MatrixVectorMultiplication::ValidationImpl() {
  return task_data->inputs_count[1] == task_data->inputs_count[2];
}

bool malyshev_v_lent_horizontal_seq::MatrixVectorMultiplication::RunImpl() {
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      result_[i] += matrix_[i * cols_ + j] * vector_[j];
    }
  }
  return true;
}

bool malyshev_v_lent_horizontal_seq::MatrixVectorMultiplication::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(result_.begin(), result_.end(), output_ptr);
  return true;
}

}  // namespace malyshev_v_lent_horizontal_seq