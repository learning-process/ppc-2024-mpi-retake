#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <vector>

#include "seq/ersoz_b_horizontal_a_vertical_b/include/ops_seq.hpp"

TEST(ersoz_b_horizontal_a_vertical_b_seq, test_pipeline_run) {
  std::size_t a_rows = 200;
  std::size_t a_cols = 150;
  auto matrix1 = GetRandomMatrix(a_rows, a_cols);
  auto matrix2 = GetRandomMatrix(a_cols, a_rows);
  constexpr int kIterations = 100;
  double total_time = 0.0;
  std::vector<int> result;
  for (int i = 0; i < kIterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    result = GetParallelOperations(matrix1, matrix2, a_rows, a_cols);
    auto end = std::chrono::high_resolution_clock::now();
    total_time += std::chrono::duration<double>(end - start).count();
  }
  double avg_time = total_time / kIterations;
  std::cout << "test_pipeline_run - Average Parallel Multiplication Time: " << avg_time << " seconds\n";
  auto expected = GetSequentialOperations(matrix1, matrix2, a_rows, a_cols, a_rows);
  ASSERT_EQ(expected, result);
  SUCCEED();
}

TEST(ersoz_b_horizontal_a_vertical_b_seq, test_task_run) {
  std::size_t a_rows = 200;
  std::size_t a_cols = 150;
  std::size_t b_cols = a_rows;
  auto matrix1 = GetRandomMatrix(a_rows, a_cols);
  auto matrix2 = GetRandomMatrix(a_cols, a_rows);

  auto start_seq = std::chrono::high_resolution_clock::now();
  auto result_seq = GetSequentialOperations(matrix1, matrix2, a_rows, a_cols, b_cols);
  auto end_seq = std::chrono::high_resolution_clock::now();
  double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();

  auto start_par = std::chrono::high_resolution_clock::now();
  auto result_par = GetParallelOperations(matrix1, matrix2, a_rows, a_cols);
  auto end_par = std::chrono::high_resolution_clock::now();
  double par_time = std::chrono::duration<double>(end_par - start_par).count();

  std::cout << "test_task_run - Sequential Time: " << seq_time << " seconds, Parallel Time: " << par_time
            << " seconds\n";
  ASSERT_EQ(result_seq, result_par);
  SUCCEED();
}
