#include "seq/strakhov_a_m_gauss_jordan/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool strakhov_a_m_gauss_jordan_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0);
  return true;
}

bool strakhov_a_m_gauss_jordan_seq::TestTaskSequential::ValidationImpl() {
  if (task_data->inputs_count[1] == 0) {
    return false;
  }
  if (task_data->inputs_count[0] != (task_data->inputs_count[1] + 1)) {
    return false;
  }
  if (task_data->inputs_count[1] != task_data->outputs_count[0]) {
    return false;
  }
  row_size_ = task_data->inputs_count[0];
  col_size_ = task_data->inputs_count[1];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(in_ptr, in_ptr + (row_size_ * col_size_));

  for (size_t i = 0; i < col_size_; i++) {
    bool flag1 = true;
    bool flag2 = true;
    for (size_t j = 0; j < col_size_; j++) {
      if (flag1 && (input_[(j * row_size_) + i] != 0)) {
        flag1 = false;
        if (!flag2) {
          break;
        }
      }
      if (flag2 && (input_[(i * row_size_) + j] != 0)) {
        flag2 = false;
        if (!flag1) {
          break;
        }
      }
    }
    if (flag1 || flag2) {
      return false;
    }
  }
  return true;
}

bool strakhov_a_m_gauss_jordan_seq::TestTaskSequential::RunImpl() {
  for (size_t i = 0; i < col_size_; i++) {
    if (input_[(i * row_size_) + i] == 0) {
      size_t k = i + 1;
      while ((k < col_size_) && (input_[(k * row_size_) + i] == 0)) {
        k += 1;
      }
      if (k == col_size_) {
        return false;
      }
      for (size_t j = 0; j < row_size_; j++) {
        input_[(i * row_size_) + j] = input_[k * row_size_ + j];
      }
    }
    if (input_[(i * row_size_) + i] != 1.0) {
      for (size_t j = i + 1; j < row_size_; j++) {
        input_[(i * row_size_) + j] /= input_[(i * row_size_) + i];
      }
      input_[(i * row_size_) + i] = 1.0;
    }
    if (input_[(i * row_size_) + i] != 1.0) {
      for (size_t j = i + 1; j < row_size_; j++) {
        input_[(i * row_size_) + j] /= input_[(i * row_size_) + i];
      }
      input_[(i * row_size_) + i] = 1.0;
    }
    for (size_t k = 0; k < col_size_; k++) {
      if (i == k) {
        continue;
      };
      double kf = input_[(k * row_size_) + i] / input_[(i * row_size_) + i];
      for (size_t j = i + 1; j < row_size_; j++) {
        input_[k * row_size_ + j] -= (input_[row_size_ * i + j] * kf);
      }
      input_[(k * row_size_) + i] = 0;
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
