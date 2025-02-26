#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "seq/ersoz_b_horizontal_a_vertical_b/include/ops_seq.hpp"

TEST(ersoz_b_horizontal_a_vertical_b_seq, test_pipeline_run) {
  // This performance test measures the average time for parallel multiplication
  // over multiple iterations.
  std::size_t A_rows = 200, A_cols = 150;
  // Matrix2 is assumed to have dimensions A_cols x A_rows so that the product is A_rows x A_rows.
  auto matrix1 = getRandomMatrix(A_rows, A_cols);
  auto matrix2 = getRandomMatrix(A_cols, A_rows);
  constexpr int iterations = 100;
  double total_time = 0.0;
  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = getParallelOperations(matrix1, matrix2, A_rows, A_cols);
    auto end = std::chrono::high_resolution_clock::now();
    total_time += std::chrono::duration<double>(end - start).count();
  }
  double avg_time = total_time / iterations;
  std::cout << "test_pipeline_run - Average Parallel Multiplication Time: " << avg_time << " seconds\n";
  SUCCEED();
}

TEST(ersoz_b_horizontal_a_vertical_b_seq, test_task_run) {
  // This test compares the running time of sequential and parallel multiplication.
  std::size_t A_rows = 200, A_cols = 150;
  std::size_t B_cols = A_rows;  // Result dimensions: A_rows x A_rows.
  auto matrix1 = getRandomMatrix(A_rows, A_cols);
  auto matrix2 = getRandomMatrix(A_cols, A_rows);

  auto start_seq = std::chrono::high_resolution_clock::now();
  auto result_seq = getSequentialOperations(matrix1, matrix2, A_rows, A_cols, B_cols);
  auto end_seq = std::chrono::high_resolution_clock::now();
  double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();

  auto start_par = std::chrono::high_resolution_clock::now();
  auto result_par = getParallelOperations(matrix1, matrix2, A_rows, A_cols);
  auto end_par = std::chrono::high_resolution_clock::now();
  double par_time = std::chrono::duration<double>(end_par - start_par).count();

  std::cout << "test_task_run - Sequential Time: " << seq_time << " seconds, Parallel Time: " << par_time
            << " seconds\n";

  ASSERT_EQ(result_seq, result_par);
  SUCCEED();
}
