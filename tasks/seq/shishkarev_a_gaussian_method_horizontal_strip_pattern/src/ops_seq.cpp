#include "seq/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

int shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::matrix_rank(int n, int m, std::vector<double> a) {
  const double EPS = 1e-6;

  int rank = m;
  for (int i = 0; i < m; ++i) {
    int j;
    for (j = 0; j < n; ++j) {
      if (std::abs(a[j * n + i]) > EPS) {
        break;
      }
    }
    if (j == n) {
      --rank;
    } else {
      for (int k = i + 1; k < m; ++k) {
        double ml = a[k * n + i] / a[i * n + i];
        for (j = i; j < n - 1; ++j) {
          a[k * n + j] -= a[i * n + j] * ml;
        }
      }
    }
  }
  return rank;
}

int shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::determinant(int n, int m, std::vector<double> a) {
  const double EPS = 1e-6;
  double det = 1;

  for (int i = 0; i < m; ++i) {
    int idx = i;
    for (int k = i + 1; k < m; ++k) {
      if (std::abs(a[k * n + i]) > std::abs(a[idx * n + i])) {
        idx = k;
      }
    }
    if (std::abs(a[idx * n + i]) < EPS) {
      return 0;
    }
    if (idx != i) {
      for (int j = 0; j < n - 1; ++j) {
        double tmp = a[i * n + j];
        a[i * n + j] = a[idx * n + j];
        a[idx * n + j] = tmp;
      }
      det *= -1;
    }
    det *= a[i * n + i];
    for (int k = i + 1; k < m; ++k) {
      double ml = a[k * n + i] / a[i * n + i];
      for (int j = i; j < n - 1; ++j) {
        a[k * n + j] -= a[i * n + j] * ml;
      }
    }
  }
  return det;
}

template <class InOutType>
bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<InOutType>::PreProcessingImpl() {

  matrix = std::vector<double>(task_data->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], matrix.begin());
  cols = task_data->inputs_count[1];
  rows = task_data->inputs_count[2];

  res = std::vector<double>(cols - 1, 0);
  return true;
}

template <typename InOutType>
bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<InOutType>::ValidationImpl() {

  matrix = std::vector<double>(task_data->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], matrix.begin());
  cols = task_data->inputs_count[1];
  rows = task_data->inputs_count[2];

  return task_data->inputs_count[0] > 1 && rows == cols - 1 && determinant(cols, rows, matrix) != 0 &&
         matrix_rank(cols, rows, matrix) == rows;
}

template <typename InOutType>
bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<InOutType>::RunImpl() {

  for (int i = 0; i < rows - 1; ++i) {
    for (int k = i + 1; k < rows; ++k) {
      double m = matrix[k * cols + i] / matrix[i * cols + i];
      for (int j = i; j < cols; ++j) {
        matrix[k * cols + j] -= matrix[i * cols + j] * m;
      }
    }
  }
  for (int i = rows - 1; i >= 0; --i) {
    double sum = matrix[i * cols + rows];
    for (int j = i + 1; j < cols - 1; ++j) {
      sum -= matrix[i * cols + j] * res[j];
    }
    res[i] = sum / matrix[i * cols + i];
  }
  return true;
}

template <typename InOutType>
bool shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<InOutType>::PostProcessingImpl() {

  auto* this_matrix = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(res.begin(), res.end(), this_matrix);
  return true;
}

template class shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<int32_t>;
template class shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<float>;
template class shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<double>;
template class shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<uint8_t>;
template class shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<int64_t>;