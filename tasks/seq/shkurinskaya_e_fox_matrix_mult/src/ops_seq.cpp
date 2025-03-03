#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "seq/shkurinskaya_e_fox_matrix_mult/include/ops_sec.hpp"

std::vector<double> shkurinskaya_e_fox_mat_mul_seq::GetRandomMatrix(int rows, int cols) {
  std::vector<double> result(rows * cols);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-50.0, 50.0);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result[i * cols + j] = dis(gen);
    }
  }

  return result;
}

bool shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential::PreProcessingImpl() {
  matrix_size_ = (int)(task_data->inputs_count[0]);

  inputA_.resize(matrix_size_ * matrix_size_);
  inputB_.resize(matrix_size_ * matrix_size_);
  output_.resize(matrix_size_ * matrix_size_);

  auto *it1 = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *it2 = reinterpret_cast<double *>(task_data->inputs[1]);
  std::copy(it1, it1 + (matrix_size_ * matrix_size_), inputA_.data());
  std::copy(it2, it2 + (matrix_size_ * matrix_size_), inputB_.data());
  return true;
}

bool shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential::ValidationImpl() {
  return (task_data->inputs_count[0] > 0) && (task_data->outputs_count[0] == task_data->inputs_count[0]);
}

bool shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential::RunImpl() {
  for (int i = 0; i < matrix_size_; ++i) {
    for (int j = 0; j < matrix_size_; ++j) {
      output_[(i * matrix_size_) + j] = 0.0;
      for (int k = 0; k < matrix_size_; ++k) {
        output_[(i * matrix_size_) + j] += inputA_[(i * matrix_size_) + k] * inputB_[(k * matrix_size_) + j];
      }
    }
  }
  return true;
}

bool shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential::PostProcessingImpl() {
  auto *it1 = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(output_, it1);
  return true;
}
