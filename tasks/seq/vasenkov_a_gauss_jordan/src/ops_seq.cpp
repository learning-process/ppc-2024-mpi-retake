#include "seq/vasenkov_a_gauss_jordan/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool vasenkov_a_gauss_jordan_seq::GaussJordanMethodSequential::ValidationImpl() {
  int n_val = *reinterpret_cast<int *>(task_data->inputs[1]);
  int matrix_size = static_cast<int>(task_data->inputs_count[0]);
  return n_val * (n_val + 1) == matrix_size;
}

bool vasenkov_a_gauss_jordan_seq::GaussJordanMethodSequential::PreProcessingImpl() {
  auto *matrix_data = reinterpret_cast<double *>(task_data->inputs[0]);
  int matrix_size = static_cast<int>(task_data->inputs_count[0]);
  n_size_ = *reinterpret_cast<int *>(task_data->inputs[1]);
  sys_matrix_.assign(matrix_data, matrix_data + matrix_size);
  return true;
}

bool vasenkov_a_gauss_jordan_seq::GaussJordanMethodSequential::RunImpl() {
  for (int k = 0; k < n_size_; ++k) {
    if (sys_matrix_[(k * (n_size_ + 1)) + k] == 0.0) {
      int swap_row = -1;
      for (int i = k + 1; i < n_size_; ++i) {
        if (std::abs(sys_matrix_[(i * (n_size_ + 1)) + k]) > 1e-6) {
          swap_row = i;
          break;
        }
      }
      if (swap_row == -1) return false;
      for (int col = 0; col <= n_size_; ++col) {
        std::swap(sys_matrix_[(k * (n_size_ + 1)) + col], sys_matrix_[(swap_row * (n_size_ + 1)) + col]);
      }
    }
    const double pivot = sys_matrix_[(k * (n_size_ + 1)) + k];
    for (int j = k; j <= n_size_; ++j) {
      sys_matrix_[(k * (n_size_ + 1)) + j] /= pivot;
    }
    for (int i = 0; i < n_size_; ++i) {
      if (i != k && sys_matrix_[(i * (n_size_ + 1)) + k] != 0.0) {
        const double factor = sys_matrix_[(i * (n_size_ + 1)) + k];
        for (int j = k; j <= n_size_; ++j) {
          sys_matrix_[(i * (n_size_ + 1)) + j] -= factor * sys_matrix_[(k * (n_size_ + 1)) + j];
        }
        sys_matrix_[(i * (n_size_ + 1)) + k] = 0.0;
      }
    }
  }

  return true;
}

bool vasenkov_a_gauss_jordan_seq::GaussJordanMethodSequential::PostProcessingImpl() {
  auto *output_data = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(sys_matrix_, output_data);
  return true;
}
