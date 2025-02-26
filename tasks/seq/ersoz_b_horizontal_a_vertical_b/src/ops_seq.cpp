#include "seq/ersoz_b_horizontal_a_vertical_b/include/ops_seq.hpp"

#include <random>
#include <thread>
#include <vector>

std::vector<int> GetRandomMatrix(std::size_t row_count, std::size_t column_count) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<int> matrix(row_count * column_count);
  for (std::size_t i = 0; i < row_count; ++i) {
    for (std::size_t j = 0; j < column_count; ++j) {
      matrix[i * column_count + j] = gen() % 100;
    }
  }
  return matrix;
}

std::vector<int> GetSequentialOperations(const std::vector<int>& matrix1, const std::vector<int>& matrix2,
                                         std::size_t a_rows, std::size_t a_cols, std::size_t b_cols) {
  std::vector<int> result(a_rows * b_cols, 0);
  for (std::size_t i = 0; i < a_rows; ++i) {
    for (std::size_t j = 0; j < b_cols; ++j) {
      int sum = 0;
      for (std::size_t k = 0; k < a_cols; ++k) {
        sum += matrix1[i * a_cols + k] * matrix2[k * b_cols + j];
      }
      result[i * b_cols + j] = sum;
    }
  }
  return result;
}

std::vector<int> GetParallelOperations(const std::vector<int>& matrix1, const std::vector<int>& matrix2,
                                       std::size_t a_rows, std::size_t a_cols) {
  const std::size_t b_cols = a_rows;
  std::vector<int> result(a_rows * b_cols, 0);
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 2;
  }
  auto worker = [&](std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) {
      for (std::size_t j = 0; j < b_cols; ++j) {
        int sum = 0;
        for (std::size_t k = 0; k < a_cols; ++k) {
          sum += matrix1[i * a_cols + k] * matrix2[k * b_cols + j];
        }
        result[i * b_cols + j] = sum;
      }
    }
  };
  std::vector<std::thread> threads;
  std::size_t rows_per_thread = a_rows / num_threads;
  std::size_t extra = a_rows % num_threads;
  std::size_t current = 0;
  for (unsigned int t = 0; t < num_threads; ++t) {
    std::size_t start = current;
    std::size_t end = start + rows_per_thread + (t < extra ? 1 : 0);
    threads.emplace_back(worker, start, end);
    current = end;
  }
  for (auto& th : threads) {
    th.join();
  }
  return result;
}
