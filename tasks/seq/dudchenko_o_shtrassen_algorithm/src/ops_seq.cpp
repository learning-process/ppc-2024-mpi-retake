#include "seq/dudchenko_o_shtrassen_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

bool dudchenko_o_shtrassen_algorithm_seq::StrassenAlgoriphmSequential::PreProcessingImpl() {
  auto* inputs_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* inputs_b = reinterpret_cast<double*>(task_data->inputs[1]);

  size_ = static_cast<size_t>(std::sqrt(task_data->inputs_count[0]));
  matrixA_.assign(inputs_a, inputs_a + (size_ * size_));
  matrixB_.assign(inputs_b, inputs_b + (size_ * size_));
  result_.resize(size_ * size_);

  return true;
}

bool dudchenko_o_shtrassen_algorithm_seq::StrassenAlgoriphmSequential::ValidationImpl() {
  return !task_data->inputs.empty() && task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->inputs_count[0] == static_cast<size_t>(std::sqrt(task_data->inputs_count[0])) *
                                           static_cast<size_t>(std::sqrt(task_data->inputs_count[0])) &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool dudchenko_o_shtrassen_algorithm_seq::StrassenAlgoriphmSequential::RunImpl() {
  result_ = dudchenko_o_shtrassen_algorithm_seq::Strassen(matrixA_, matrixB_, size_);
  return true;
}

bool dudchenko_o_shtrassen_algorithm_seq::StrassenAlgoriphmSequential::PostProcessingImpl() {
  auto* outputs = reinterpret_cast<double*>(task_data->outputs[0]);

  std::copy(result_.begin(), result_.end(), outputs);  // NOLINT

  return true;
}

std::vector<double> dudchenko_o_shtrassen_algorithm_seq::Add(const std::vector<double>& a, const std::vector<double>& b,
                                                             size_t n) {
  std::vector<double> result(n * n);
  std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<double>());  // NOLINT
  return result;
}

std::vector<double> dudchenko_o_shtrassen_algorithm_seq::Subtract(const std::vector<double>& a,
                                                                  const std::vector<double>& b, size_t n) {
  std::vector<double> result(n * n);
  std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<double>());  // NOLINT
  return result;
}

std::vector<double> dudchenko_o_shtrassen_algorithm_seq::Strassen(const std::vector<double>& a,
                                                                  const std::vector<double>& b, size_t n) {
  if (n == 1) {
    return {a[0] * b[0]};
  }

  size_t new_size = 1;
  while (new_size < n) {
    new_size *= 2;
  }

  std::vector<double> a_ext(new_size * new_size, 0.0);
  std::vector<double> b_ext(new_size * new_size, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      a_ext[(i * new_size) + j] = a[(i * n) + j];
      b_ext[(i * new_size) + j] = b[(i * n) + j];
    }
  }

  size_t half = new_size / 2;
  size_t half_squared = half * half;
  auto get_submatrix = [&](const std::vector<double>& m, size_t row, size_t col) {
    std::vector<double> sub(half_squared);
    for (size_t i = 0; i < half; ++i) {
      for (size_t j = 0; j < half; ++j) {
        sub[(i * half) + j] = m[((i + row) * new_size) + j + col];
      }
    }
    return sub;
  };

  auto a11 = get_submatrix(a_ext, 0, 0);
  auto a12 = get_submatrix(a_ext, 0, half);
  auto a21 = get_submatrix(a_ext, half, 0);
  auto a22 = get_submatrix(a_ext, half, half);
  auto b11 = get_submatrix(b_ext, 0, 0);
  auto b12 = get_submatrix(b_ext, 0, half);
  auto b21 = get_submatrix(b_ext, half, 0);
  auto b22 = get_submatrix(b_ext, half, half);

  auto m1 = Strassen(Add(a11, a22, half), Add(b11, b22, half), half);
  auto m2 = Strassen(Add(a21, a22, half), b11, half);
  auto m3 = Strassen(a11, Subtract(b12, b22, half), half);
  auto m4 = Strassen(a22, Subtract(b21, b11, half), half);
  auto m5 = Strassen(Add(a11, a12, half), b22, half);
  auto m6 = Strassen(Subtract(a21, a11, half), Add(b11, b12, half), half);
  auto m7 = Strassen(Subtract(a12, a22, half), Add(b21, b22, half), half);

  std::vector<double> result_ext(new_size * new_size, 0.0);
  for (size_t i = 0; i < half; ++i) {
    for (size_t j = 0; j < half; ++j) {
      size_t idx = (i * half) + j;
      size_t ext_idx = (i * new_size) + j;
      result_ext[ext_idx] = m1[idx] + m4[idx] - m5[idx] + m7[idx];
      result_ext[ext_idx + half] = m3[idx] + m5[idx];
      result_ext[((i + half) * new_size) + j] = m2[idx] + m4[idx];
      result_ext[((i + half) * new_size) + j + half] = m1[idx] + m3[idx] - m2[idx] + m6[idx];
    }
  }

  std::vector<double> result(n * n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      result[(i * n) + j] = result_ext[(i * new_size) + j];
    }
  }

  return result;
}