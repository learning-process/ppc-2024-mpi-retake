#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "seq/shkurinskaya_e_fox_matrix_mult/include/ops_sec.hpp"

std::vector<double> shkurinskaya_e_fox_mat_mul_seq::getRandomMatrix(int rows, int cols) {
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
  matrix_size = task_data->inputs_count[0];

  inputA.resize(matrix_size * matrix_size);
  inputB.resize(matrix_size * matrix_size);
  output.resize(matrix_size * matrix_size);

  double *it1 = (double *)(task_data->inputs[0]);
  double *it2 = (double *)(task_data->inputs[1]);
  std::copy(it1, it1 + matrix_size * matrix_size, inputA.data());
  std::copy(it2, it2 + matrix_size * matrix_size, inputB.data());
  return true;
}

bool shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential::ValidationImpl() {
  return (task_data->inputs_count[0] > 0) && (task_data->outputs_count[0] == task_data->inputs_count[0]);
}

bool shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential::RunImpl() {
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      output[i * matrix_size + j] = 0.0;
      for (int k = 0; k < matrix_size; ++k) {
        output[i * matrix_size + j] += inputA[i * matrix_size + k] * inputB[k * matrix_size + j];
      }
    }
  }
  return true;
}

bool shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential::PostProcessingImpl() {
  double *it1 = (double *)(task_data->outputs[0]);
  std::copy(output.begin(), output.end(), it1);
  return true;
}
