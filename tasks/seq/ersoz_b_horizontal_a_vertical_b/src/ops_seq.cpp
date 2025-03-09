#include "seq/ersoz_b_horizontal_a_vertical_b/include/ops_seq.hpp"

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

std::vector<int> GetRandomMatrix(std::size_t row_count, std::size_t column_count) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<int> matrix(row_count * column_count);
  for (std::size_t i = 0; i < row_count; ++i) {
    for (std::size_t j = 0; j < column_count; ++j) {
      matrix[(i * column_count) + j] = static_cast<int>(gen() % 100);
    }
  }
  return matrix;
}

std::vector<int> GetSequentialOperations(const std::vector<int>& matrix1, const std::vector<int>& matrix2,
                                         std::size_t a_rows, std::size_t a_cols, std::size_t b_cols) {
  if (matrix1.size() != a_rows * a_cols) {
    throw std::invalid_argument("Invalid dimensions for matrix1");
  }
  if (matrix2.size() != a_cols * b_cols) {
    throw std::invalid_argument("Invalid dimensions for matrix2");
  }
  std::vector<int> result(a_rows * b_cols, 0);
  for (std::size_t i = 0; i < a_rows; ++i) {
    for (std::size_t j = 0; j < b_cols; ++j) {
      int sum = 0;
      for (std::size_t k = 0; k < a_cols; ++k) {
        sum += matrix1[(i * a_cols) + k] * matrix2[(k * b_cols) + j];
      }
      result[(i * b_cols) + j] = sum;
    }
  }
  return result;
}

std::vector<int> GetParallelOperations(const std::vector<int>& matrix1, const std::vector<int>& matrix2,
                                       std::size_t a_rows, std::size_t a_cols) {
  const std::size_t b_cols = a_rows;
  if (matrix1.size() != a_rows * a_cols) {
    throw std::invalid_argument("Invalid dimensions for matrix1");
  }
  if (matrix2.size() != a_cols * b_cols) {
    throw std::invalid_argument("Invalid dimensions for matrix2");
  }
  return GetSequentialOperations(matrix1, matrix2, a_rows, a_cols, b_cols);
}
