#include "seq/Konstantinov_I_Gauss_Jordan_method/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq::PreProcessingImpl() {
  n = *reinterpret_cast<int*>(task_data->inputs[0]);
  matrix = std::vector<double>(reinterpret_cast<double*>(task_data->inputs[1]),
                               reinterpret_cast<double*>(task_data->inputs[1]) + n * (n + 1));
  solution = std::vector<double>(n, 0.0);
  return true;
}

bool konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq::ValidationImpl() {
  int numRows = task_data->inputs_count[0];
  int numCols = (task_data->inputs_count[0] > 0) ? (numRows + 1) : 0;
  if (numRows <= 0 || numCols <= 0) {
    return false;
  }
  auto expectedSize = static_cast<size_t>(numRows * numCols);
  if (task_data->inputs_count[1] != expectedSize) {
    return false;
  }
  auto* matrixData = reinterpret_cast<double*>(task_data->inputs[1]);
  for (int i = 0; i < numRows; ++i) {
    auto value = matrixData[i * numCols + i];
    if (value == 0.0) {
      return false;
    }
  }
  return true;
}

bool konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq::RunImpl() {
  for (int k = 0; k < n; ++k) {
    int max_row = k;
    for (int i = k + 1; i < n; ++i) {
      if (std::abs(matrix[i * (n + 1) + k]) > std::abs(matrix[max_row * (n + 1) + k])) {
        max_row = i;
      }
    }
    if (max_row != k) {
      for (int j = k; j <= n; ++j) {
        std::swap(matrix[k * (n + 1) + j], matrix[max_row * (n + 1) + j]);
      }
    }
    double diag = matrix[k * (n + 1) + k];
    for (int j = k; j <= n; ++j) {
      matrix[k * (n + 1) + j] /= diag;
    }
    for (int i = k + 1; i < n; ++i) {
      double factor = matrix[i * (n + 1) + k];
      for (int j = k; j <= n; ++j) {
        matrix[i * (n + 1) + j] -= matrix[k * (n + 1) + j] * factor;
      }
    }
  }
  for (int k = n - 1; k >= 0; --k) {
    for (int i = k - 1; i >= 0; --i) {
      double factor = matrix[i * (n + 1) + k];
      for (int j = k; j <= n; ++j) {
        matrix[i * (n + 1) + j] -= matrix[k * (n + 1) + j] * factor;
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    solution[i] = matrix[i * (n + 1) + n];
  }

  return true;
}

bool konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq::PostProcessingImpl() {
  for (int i = 0; i < n; ++i) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = solution[i];
  }
  return true;
}