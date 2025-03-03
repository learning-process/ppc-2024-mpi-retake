#include "seq/strakhov_a_m_gauss_jordan/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

bool checkZero(size_t col_size, size_t row_size, std::vector<double>& input) {
  for (size_t i = 0; i < col_size; i++) {
    bool flag1 = true;
    bool flag2 = true;
    for (size_t j = 0; (j < col_size) && (flag1 || flag2); j++) {
      flag1 = flag1 && (input[(j * row_size) + i] == 0);
      flag2 = flag2 && (input[(i * row_size) + j] == 0);
    }
    if (flag1 || flag2) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool strakhov_a_m_gauss_jordan_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0);
  return true;
}

bool strakhov_a_m_gauss_jordan_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  if (task_data->inputs_count[1] == 0) return false;
  if (task_data->inputs_count[0] != (task_data->inputs_count[1] + 1)) {
    return false;
  }
  if (task_data->inputs_count[1] != task_data->outputs_count[0]) {
    return false;
  }
  row_size_ = task_data->inputs_count[0];
  col_size_ = task_data->inputs_count[1];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + row_size_ * col_size_);
  return checkZero(col_size_, row_size_, input_);
}

bool strakhov_a_m_gauss_jordan_seq::TestTaskSequential::RunImpl() {
  // Multiply matrices
  for (size_t i = 0; i < col_size_; i++) {
    size_t i_row = i * row_size_;
    if (input_[i_row + i] != 1.0) {
      for (size_t j = i + 1; j < row_size_; j++) {
        input_[i_row + j] /= input_[i_row + i];
      }
      input_[i_row + i] = 1.0;
    }
    for (size_t k = 0; k < col_size_; k++) {
      if (i == k) {
        continue;
      };
      size_t k_row = k * row_size_;
      double kf = input_[k_row + i] / input_[i_row + i];
      for (size_t j = i + 1; j < row_size_; j++) {
        input_[k_row + j] -= (input_[row_size_ * i + j] * kf);
      }
      input_[k_row + i] = 0;
    }
  }
  return true;
}

bool strakhov_a_m_gauss_jordan_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = input_[(i + 1) * row_size_ - 1];
  }
  return true;
}
