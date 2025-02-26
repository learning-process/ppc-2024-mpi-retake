#include <gtest/gtest.h>

#include <vector>

#include "ersoz_b_horizontal_a_vertical_b.hpp"

TEST(Generation_Matrix, can_generate_square_matrix) {
  auto mat = getRandomMatrix(10, 10);
  ASSERT_EQ(mat.size(), 100);
}

TEST(Generation_Matrix, can_generate_arbitrary_matrix) {
  auto mat = getRandomMatrix(10, 15);
  ASSERT_EQ(mat.size(), 150);
}

TEST(Sequential_Operations_SEQ, getSequentialOperations_can_work_with_square_matrix) {
  std::vector<int> matrix1 = getRandomMatrix(10, 10);
  std::vector<int> matrix2 = getRandomMatrix(10, 10);
  auto res = getSequentialOperations(matrix1, matrix2, 10, 10, 10);
  ASSERT_EQ(res.size(), 100);
}

TEST(Sequential_Operations_SEQ, getSequentialOperations_can_work_with_arbitrary_matrix) {
  std::vector<int> matrix1 = getRandomMatrix(10, 15);
  std::vector<int> matrix2 = getRandomMatrix(15, 10);
  auto res = getSequentialOperations(matrix1, matrix2, 10, 15, 10);
  ASSERT_EQ(res.size(), 100);
}

TEST(Sequential_Operations_SEQ, getSequentialOperations_works_correctly_with_square_matrix) {
  std::vector<int> matrix1 = {1, 1, 1, 1};
  std::vector<int> matrix2 = {2, 2, 2, 2};
  std::vector<int> expected = {4, 4, 4, 4};
  auto res = getSequentialOperations(matrix1, matrix2, 2, 2, 2);
  ASSERT_EQ(res, expected);
}

TEST(Sequential_Operations_SEQ, getSequentialOperations_works_correctly_with_arbitrary_matrix) {
  std::vector<int> matrix1 = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> matrix2 = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int> expected = {8, 8, 8, 8, 8, 8};
  auto res = getSequentialOperations(matrix1, matrix2, 2, 4, 3);
  ASSERT_EQ(res, expected);
}

TEST(Parallel_Operations_SEQ, getParallelOperations_can_work_with_square_matrix) {
  std::vector<int> matrix1 = getRandomMatrix(20, 20);
  std::vector<int> matrix2 = getRandomMatrix(20, 20);
  auto res = getParallelOperations(matrix1, matrix2, 20, 20);
  ASSERT_EQ(res.size(), 20 * 20);
}

TEST(Parallel_Operations_SEQ, getParallelOperations_can_work_with_arbitrary_matrix) {
  // For arbitrary: matrix1 is 20x30 and matrix2 is 30x20, so the result is 20x20.
  std::vector<int> matrix1 = getRandomMatrix(20, 30);
  std::vector<int> matrix2 = getRandomMatrix(30, 20);
  auto res = getParallelOperations(matrix1, matrix2, 20, 30);
  ASSERT_EQ(res.size(), 20 * 20);
}

TEST(Parallel_Operations_SEQ, getParallelOperations_works_correctly_with_square_matrix) {
  std::size_t rows = 20, cols = 20;
  auto matrix1 = getRandomMatrix(rows, cols);
  auto matrix2 = getRandomMatrix(cols, rows);
  auto res_parallel = getParallelOperations(matrix1, matrix2, rows, cols);
  auto res_sequential = getSequentialOperations(matrix1, matrix2, rows, cols, rows);
  ASSERT_EQ(res_parallel, res_sequential);
}

TEST(Parallel_Operations_SEQ, getParallelOperations_works_correctly_with_more_columns) {
  std::size_t rows = 20, cols = 30;
  auto matrix1 = getRandomMatrix(rows, cols);
  auto matrix2 = getRandomMatrix(cols, rows);
  auto res_parallel = getParallelOperations(matrix1, matrix2, rows, cols);
  auto res_sequential = getSequentialOperations(matrix1, matrix2, rows, cols, rows);
  ASSERT_EQ(res_parallel, res_sequential);
}

TEST(Parallel_Operations_SEQ, getParallelOperations_works_correctly_with_more_rows) {
  std::size_t rows = 30, cols = 20;
  auto matrix1 = getRandomMatrix(rows, cols);
  auto matrix2 = getRandomMatrix(cols, rows);
  auto res_parallel = getParallelOperations(matrix1, matrix2, rows, cols);
  auto res_sequential = getSequentialOperations(matrix1, matrix2, rows, cols, rows);
  ASSERT_EQ(res_parallel, res_sequential);
}

TEST(Parallel_Operations_SEQ, getParallelOperations_works_correctly_with_odd_size_and_more_rows) {
  std::size_t rows = 27, cols = 15;
  auto matrix1 = getRandomMatrix(rows, cols);
  auto matrix2 = getRandomMatrix(cols, rows);
  auto res_parallel = getParallelOperations(matrix1, matrix2, rows, cols);
  auto res_sequential = getSequentialOperations(matrix1, matrix2, rows, cols, rows);
  ASSERT_EQ(res_parallel, res_sequential);
}

TEST(Parallel_Operations_SEQ, getParallelOperations_works_correctly_with_odd_size_and_more_columns) {
  std::size_t rows = 15, cols = 27;
  auto matrix1 = getRandomMatrix(rows, cols);
  auto matrix2 = getRandomMatrix(cols, rows);
  auto res_parallel = getParallelOperations(matrix1, matrix2, rows, cols);
  auto res_sequential = getSequentialOperations(matrix1, matrix2, rows, cols, rows);
  ASSERT_EQ(res_parallel, res_sequential);
}
