#include "seq/Konstantinov_I_Gauss_Jordan_method/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq::PreProcessingImpl() {
  n_ = *reinterpret_cast<int*>(task_data->inputs[0]);
  matrix_ = std::vector<double>(reinterpret_cast<double*>(task_data->inputs[1]),
                                reinterpret_cast<double*>(task_data->inputs[1]) + (n_ * (n_ + 1)));
  solution_ = std::vector<double>(n_, 0.0);
  return true;
}

bool konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq::ValidationImpl() {
  int num_rows = static_cast<int>(task_data->inputs_count[0]);
  int num_cols = (task_data->inputs_count[0] > 0) ? (num_rows + 1) : 0;
  if (num_rows <= 0 || num_cols <= 0) {
    return false;
  }
  auto expected_size = static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols);
  if (task_data->inputs_count[1] != expected_size) {
    return false;
  }
  auto* matrix_data = reinterpret_cast<double*>(task_data->inputs[1]);
  for (int i = 0; i < num_rows; ++i) {
    auto value = matrix_data[(i * num_cols) + i];
    if (value == 0.0) {
      return false;
    }
  }
  return true;
}

bool konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq::RunImpl() {
  for (int k = 0; k < n_; ++k) {
    int max_row = k;
    for (int i = k + 1; i < n_; ++i) {
      if (std::abs(matrix_[(i * (n_ + 1)) + k]) > std::abs(matrix_[(max_row * (n_ + 1)) + k])) {
        max_row = i;
      }
    }
    if (max_row != k) {
      for (int j = k; j <= n_; ++j) {
        std::swap(matrix_[(k * (n_ + 1)) + j], matrix_[(max_row * (n_ + 1)) + j]);
      }
    }
    double diag = matrix_[(k * (n_ + 1)) + k];
    for (int j = k; j <= n_; ++j) {
      matrix_[(k * (n_ + 1)) + j] /= diag;
    }
    for (int i = k + 1; i < n_; ++i) {
      double factor = matrix_[(i * (n_ + 1)) + k];
      for (int j = k; j <= n_; ++j) {
        matrix_[(i * (n_ + 1)) + j] -= matrix_[(k * (n_ + 1)) + j] * factor;
      }
    }
  }
  for (int k = n_ - 1; k >= 0; --k) {
    for (int i = k - 1; i >= 0; --i) {
      double factor = matrix_[(i * (n_ + 1)) + k];
      for (int j = k; j <= n_; ++j) {
        matrix_[(i * (n_ + 1)) + j] -= matrix_[(k * (n_ + 1)) + j] * factor;
      }
    }
  }
  for (int i = 0; i < n_; ++i) {
    solution_[i] = matrix_[(i * (n_ + 1)) + n_];
  }

  return true;
}

bool konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq::PostProcessingImpl() {
  for (int i = 0; i < n_; ++i) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = solution_[i];
  }
  return true;
}