#include "seq/agafeev_s_strassen_alg/include/strassen_seq.hpp"

#include <vector>

namespace {
bool IsPowerOfTwo(unsigned int n) { return (n != 0) && ((n & (n - 1)) == 0); }
}  // namespace

namespace agafeev_s_strassen_alg_seq {

std::vector<double> AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int n) {
  std::vector<double> c(n * n);
  for (int i = 0; i < n * n; ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

std::vector<double> SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b, int n) {
  std::vector<double> c(n * n);
  for (int i = 0; i < n * n; ++i) {
    c[i] = a[i] - b[i];
  }
  return c;
}

void SplitMatrix(const std::vector<double>& a, std::vector<double>& a11, std::vector<double>& a12,
                 std::vector<double>& a21, std::vector<double>& a22, int n) {
  int half = n / 2;
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      a11[(i * half) + j] = a[(i * n) + j];
      a12[(i * half) + j] = a[(i * n) + (j + half)];
      a21[(i * half) + j] = a[((i + half) * n) + j];
      a22[(i * half) + j] = a[((i + half) * n) + (j + half)];
    }
  }
}

std::vector<double> MergeMatrices(const std::vector<double>& a11, const std::vector<double>& a12,
                                  const std::vector<double>& a21, const std::vector<double>& a22, int n) {
  int half = n / 2;
  std::vector<double> a(n * n);
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      a[(i * n) + j] = a11[(i * half) + j];
      a[(i * n) + (j + half)] = a12[(i * half) + j];
      a[((i + half) * n) + j] = a21[(i * half) + j];
      a[((i + half) * n) + (j + half)] = a22[(i * half) + j];
    }
  }
  return a;
}

std::vector<double> StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, int n) {
  if (n == 1) {
    return {a[0] * b[0]};
  }

  int half = n / 2;

  std::vector<double> a11(half * half);
  std::vector<double> a12(half * half);
  std::vector<double> a21(half * half);
  std::vector<double> a22(half * half);
  std::vector<double> b11(half * half);
  std::vector<double> b12(half * half);
  std::vector<double> b21(half * half);
  std::vector<double> b22(half * half);
  SplitMatrix(a, a11, a12, a21, a22, n);
  SplitMatrix(b, b11, b12, b21, b22, n);

  auto p1 = StrassenMultiply(AddMatrices(a11, a22, half), AddMatrices(b11, b22, half), half);
  auto p2 = StrassenMultiply(AddMatrices(a21, a22, half), b11, half);
  auto p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half), half);
  auto p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half), half);
  auto p5 = StrassenMultiply(AddMatrices(a11, a12, half), b22, half);
  auto p6 = StrassenMultiply(SubtractMatrices(a21, a11, half), AddMatrices(b11, b12, half), half);
  auto p7 = StrassenMultiply(SubtractMatrices(a12, a22, half), AddMatrices(b21, b22, half), half);

  auto c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half), p5, half), p7, half);
  auto c12 = AddMatrices(p3, p5, half);
  auto c21 = AddMatrices(p2, p4, half);
  auto c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half), p2, half), p6, half);

  return MergeMatrices(c11, c12, c21, c22, n);
}

bool MultiplMatrixSequental::PreProcessingImpl() {
  first_input_.clear();
  second_input_.clear();
  auto* temp_ptr1 = reinterpret_cast<double*>(task_data->inputs[0]);
  first_input_.insert(first_input_.begin(), temp_ptr1,
                      temp_ptr1 + (task_data->inputs_count[0] * task_data->inputs_count[1]));
  auto* temp_ptr2 = reinterpret_cast<double*>(task_data->inputs[1]);
  second_input_.insert(second_input_.begin(), temp_ptr2,
                       temp_ptr2 + (task_data->inputs_count[2] * task_data->inputs_count[3]));
  size_ = task_data->inputs_count[0];

  return true;
}

bool MultiplMatrixSequental::ValidationImpl() {
  return (task_data->outputs_count[0] == task_data->inputs_count[0] * task_data->inputs_count[1] &&
          IsPowerOfTwo(task_data->inputs_count[0]) && IsPowerOfTwo(task_data->inputs_count[1]) &&
          IsPowerOfTwo(task_data->inputs_count[2]) && IsPowerOfTwo(task_data->inputs_count[3]));
}

bool MultiplMatrixSequental::RunImpl() {
  result_ = StrassenMultiply(first_input_, second_input_, size_);

  return true;
}

bool MultiplMatrixSequental::PostProcessingImpl() {
  for (unsigned int i = 0; i < task_data->outputs_count[0]; i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = result_[i];
  }
  return true;
}

}  // namespace agafeev_s_strassen_alg_seq
