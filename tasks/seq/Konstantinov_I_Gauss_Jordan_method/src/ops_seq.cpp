#include "seq/Konstantinov_I_Gauss_Jordan_method/include/ops_seq.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

std::vector<double> konstantinov_i_gauss_jordan_method_seq::ProcessMatrix(int n, int k, const std::vector<double>& matrix) {
  std::vector<double> result_vec(n * (n - k + 1));

  for (int i = 0; i < (n - k + 1); i++) {
    result_vec[i] = matrix[(n + 1) * k + k + i];
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < (n - k + 1); j++) {
      result_vec[(n - k + 1) * (i + 1) + j] = matrix[i * (n + 1) + k + j];
    }
  }

  for (int i = k + 1; i < n; i++) {
    for (int j = 0; j < (n - k + 1); j++) {
      result_vec[(n - k + 1) * i + j] = matrix[i * (n + 1) + k + j];
    }
  }

  return result_vec;
}

void konstantinov_i_gauss_jordan_method_seq::UpdateMatrix(int n, int k, std::vector<double>& matrix,
                                                       const std::vector<double>& iter_result) {
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < (n - k); j++) {
      matrix[i * (n + 1) + k + 1 + j] = iter_result[i * (n - k) + j];
    }
  }

  for (int i = k + 1; i < n; i++) {
    for (int j = 0; j < (n - k); j++) {
      matrix[i * (n + 1) + k + 1 + j] = iter_result[(i - 1) * (n - k) + j];
    }
  }

  for (int i = k + 1; i < n + 1; i++) matrix[k * (n + 1) + i] /= matrix[k * (n + 1) + k];

  for (int i = 0; i < n; i++) {
    matrix[i * (n + 1) + k] = 0;
  }

  matrix[k * (n + 1) + k] = 1;
}

bool konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq::ValidationImpl() {

  int n_val = *reinterpret_cast<int*>(task_data->inputs[1]);
  int matrix_size = task_data->inputs_count[0];
  return n_val * (n_val + 1) == matrix_size;
}

bool konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq::PreProcessingImpl() {

  auto* matrix_data = reinterpret_cast<double*>(task_data->inputs[0]);
  int matrix_size = task_data->inputs_count[0];
  n = *reinterpret_cast<int*>(task_data->inputs[1]);
  matrix.assign(matrix_data, matrix_data + matrix_size);

  return true;
}

bool konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq::RunImpl() {

  for (int k = 0; k < n; k++) {
    if (matrix[k * (n + 1) + k] == 0) {
      int change;
      for (change = k + 1; change < n; change++) {
        if (matrix[change * (n + 1) + k] != 0) {
          for (int col = 0; col < (n + 1); col++) {
            std::swap(matrix[k * (n + 1) + col], matrix[change * (n + 1) + col]);
          }
          break;
        }
      }
      if (change == n) return false;
    }

    std::vector<double> iter_matrix = konstantinov_i_gauss_jordan_method_seq::ProcessMatrix(n, k, matrix);

    std::vector<double> iter_result((n - 1) * (n - k));

    int ind = 0;
    for (int i = 1; i < n; ++i) {
      for (int j = 1; j < n - k + 1; ++j) {
        double rel = iter_matrix[0];
        double nel = iter_matrix[i * (n - k + 1) + j];
        double a = iter_matrix[j];
        double b = iter_matrix[i * (n - k + 1)];
        double res = nel - (a * b) / rel;
        iter_result[ind++] = res;
      }
    }

    konstantinov_i_gauss_jordan_method_seq::UpdateMatrix(n, k, matrix, iter_result);
  }

  return true;
}

bool konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq::PostProcessingImpl() {
 
  auto* output_data = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(matrix.begin(), matrix.end(), output_data);

  return true;
}