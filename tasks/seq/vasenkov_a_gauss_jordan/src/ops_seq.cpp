#include "seq/vasenkov_a_gauss_jordan/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
namespace vasenkov_a_gauss_jordan_seq {
bool GaussJordanMethodSequential::ValidationImpl() {
  int n_val = *reinterpret_cast<int *>(task_data->inputs[1]);
  int matrix_size = static_cast<int>(task_data->inputs_count[0]);
  return n_val * (n_val + 1) == matrix_size;
}

bool GaussJordanMethodSequential::PreProcessingImpl() {
  auto *matrix_data = reinterpret_cast<double *>(task_data->inputs[0]);
  int matrix_size = static_cast<int>(task_data->inputs_count[0]);
  n_size_ = *reinterpret_cast<int *>(task_data->inputs[1]);
  sys_matrix_.assign(matrix_data, matrix_data + matrix_size);
  return true;
}

bool GaussJordanMethodSequential::RunImpl() {
  for (int k = 0; k < n_size_; ++k) {
    if (!EnsureNonZeroPivot(k)) {
      return false;
    }
    NormalizeRow(k);
    EliminateColumn(k);
  }
  return true;
}

bool GaussJordanMethodSequential::EnsureNonZeroPivot(int k) {
  if (sys_matrix_[(k * (n_size_ + 1)) + k] == 0.0) {
    int swap_row = FindSwapRow(k);
    if (swap_row == -1) {
      return false;
    }
    SwapRows(k, swap_row);
  }
  return true;
}

int GaussJordanMethodSequential::FindSwapRow(int k) {
  for (int i = k + 1; i < n_size_; ++i) {
    if (std::abs(sys_matrix_[(i * (n_size_ + 1)) + k]) > 1e-6) {
      return i;
    }
  }
  return -1;
}

void GaussJordanMethodSequential::SwapRows(int row1, int row2) {
  for (int col = 0; col <= n_size_; ++col) {
    std::swap(sys_matrix_[(row1 * (n_size_ + 1)) + col], sys_matrix_[(row2 * (n_size_ + 1)) + col]);
  }
}

void GaussJordanMethodSequential::NormalizeRow(int k) {
  const double pivot = sys_matrix_[(k * (n_size_ + 1)) + k];
  for (int j = k; j <= n_size_; ++j) {
    sys_matrix_[(k * (n_size_ + 1)) + j] /= pivot;
  }
}

void GaussJordanMethodSequential::EliminateColumn(int k) {
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

bool GaussJordanMethodSequential::PostProcessingImpl() {
  auto *output_data = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(sys_matrix_, output_data);
  return true;
}

}  // namespace vasenkov_a_gauss_jordan_seq