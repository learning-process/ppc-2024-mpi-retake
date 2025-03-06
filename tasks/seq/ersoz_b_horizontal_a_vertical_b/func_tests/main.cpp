#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "seq/ersoz_b_horizontal_a_vertical_b/include/ops_seq.hpp"

TEST(Generation_Matrix, can_generate_square_matrix) {
  auto mat = GetRandomMatrix(10, 10);
  ASSERT_EQ(mat.size(), 100U);
}

TEST(Generation_Matrix, can_generate_arbitrary_matrix) {
  auto mat = GetRandomMatrix(10, 15);
  ASSERT_EQ(mat.size(), 150U);
}

TEST(Sequential_Operations_SEQ, GetSequentialOperations_can_work_with_square_matrix) {
  std::vector<int> matrix1 = GetRandomMatrix(10, 10);
  std::vector<int> matrix2 = GetRandomMatrix(10, 10);
  auto res = GetSequentialOperations(matrix1, matrix2, 10, 10, 10);
  ASSERT_EQ(res.size(), 100U);
}

TEST(Sequential_Operations_SEQ, GetSequentialOperations_can_work_with_arbitrary_matrix) {
  std::vector<int> matrix1 = GetRandomMatrix(10, 15);
  std::vector<int> matrix2 = GetRandomMatrix(15, 10);
  auto res = GetSequentialOperations(matrix1, matrix2, 10, 15, 10);
  ASSERT_EQ(res.size(), 100U);
}

TEST(Sequential_Operations_SEQ, GetSequentialOperations_works_correctly_with_square_matrix) {
  std::vector<int> matrix1 = {1, 1, 1, 1};
  std::vector<int> matrix2 = {2, 2, 2, 2};
  std::vector<int> expected = {4, 4, 4, 4};
  auto res = GetSequentialOperations(matrix1, matrix2, 2, 2, 2);
  ASSERT_EQ(expected, res);
}

TEST(Sequential_Operations_SEQ, GetSequentialOperations_works_correctly_with_arbitrary_matrix) {
  std::vector<int> matrix1 = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> matrix2 = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int> expected = {8, 8, 8, 8, 8, 8};
  auto res = GetSequentialOperations(matrix1, matrix2, 2, 4, 3);
  ASSERT_EQ(expected, res);
}

TEST(Parallel_Operations_SEQ, GetParallelOperations_can_work_with_square_matrix) {
  std::vector<int> matrix1 = GetRandomMatrix(20, 20);
  std::vector<int> matrix2 = GetRandomMatrix(20, 20);
  auto res = GetParallelOperations(matrix1, matrix2, 20, 20);
  ASSERT_EQ(res.size(), 20U * 20U);
}

TEST(Parallel_Operations_SEQ, GetParallelOperations_can_work_with_arbitrary_matrix) {
  std::vector<int> matrix1 = GetRandomMatrix(20, 30);
  std::vector<int> matrix2 = GetRandomMatrix(30, 20);
  auto res = GetParallelOperations(matrix1, matrix2, 20, 30);
  ASSERT_EQ(res.size(), 20U * 20U);
}

TEST(Parallel_Operations_SEQ, GetParallelOperations_works_correctly_with_square_matrix) {
  std::size_t rows = 20;
  std::size_t cols = 20;
  auto matrix1 = GetRandomMatrix(rows, cols);
  auto matrix2 = GetRandomMatrix(cols, rows);
  auto res_parallel = GetParallelOperations(matrix1, matrix2, rows, cols);
  auto res_sequential = GetSequentialOperations(matrix1, matrix2, rows, cols, rows);
  ASSERT_EQ(res_sequential, res_parallel);
}

TEST(Parallel_Operations_SEQ, GetParallelOperations_works_correctly_with_more_columns) {
  std::size_t rows = 20;
  std::size_t cols = 30;
  auto matrix1 = GetRandomMatrix(rows, cols);
  auto matrix2 = GetRandomMatrix(cols, rows);
  auto res_parallel = GetParallelOperations(matrix1, matrix2, rows, cols);
  auto res_sequential = GetSequentialOperations(matrix1, matrix2, rows, cols, rows);
  ASSERT_EQ(res_sequential, res_parallel);
}

TEST(Parallel_Operations_SEQ, GetParallelOperations_works_correctly_with_more_rows) {
  std::size_t rows = 30;
  std::size_t cols = 20;
  auto matrix1 = GetRandomMatrix(rows, cols);
  auto matrix2 = GetRandomMatrix(cols, rows);
  auto res_parallel = GetParallelOperations(matrix1, matrix2, rows, cols);
  auto res_sequential = GetSequentialOperations(matrix1, matrix2, rows, cols, rows);
  ASSERT_EQ(res_sequential, res_parallel);
}

TEST(Parallel_Operations_SEQ, GetParallelOperations_works_correctly_with_odd_size_and_more_rows) {
  std::size_t rows = 27;
  std::size_t cols = 15;
  auto matrix1 = GetRandomMatrix(rows, cols);
  auto matrix2 = GetRandomMatrix(cols, rows);
  auto res_parallel = GetParallelOperations(matrix1, matrix2, rows, cols);
  auto res_sequential = GetSequentialOperations(matrix1, matrix2, rows, cols, rows);
  ASSERT_EQ(res_sequential, res_parallel);
}

TEST(Parallel_Operations_SEQ, GetParallelOperations_works_correctly_with_odd_size_and_more_columns) {
  std::size_t rows = 15;
  std::size_t cols = 27;
  auto matrix1 = GetRandomMatrix(rows, cols);
  auto matrix2 = GetRandomMatrix(cols, rows);
  auto res_parallel = GetParallelOperations(matrix1, matrix2, rows, cols);
  auto res_sequential = GetSequentialOperations(matrix1, matrix2, rows, cols, rows);
  ASSERT_EQ(res_sequential, res_parallel);
}

TEST(Sequential_Operations_SEQ, Throws_On_Invalid_Matrix1_Size) {
  std::vector<int> matrix1 = {1, 2, 3};
  std::vector<int> matrix2 = {1, 2, 3, 4};
  EXPECT_THROW(GetSequentialOperations(matrix1, matrix2, 2, 2, 2), std::invalid_argument);  // NOLINT
}

TEST(Sequential_Operations_SEQ, Throws_On_Invalid_Matrix2_Size) {
  std::vector<int> matrix1 = {1, 2, 3, 4};
  std::vector<int> matrix2 = {1, 2, 3};
  EXPECT_THROW(GetSequentialOperations(matrix1, matrix2, 2, 2, 2), std::invalid_argument);  // NOLINT
}

TEST(Parallel_Operations_SEQ, Throws_On_Invalid_Matrix1_Size) {
  std::vector<int> matrix1 = {1, 2, 3};
  std::vector<int> matrix2 = {1, 2, 3, 4, 5, 6};
  EXPECT_THROW(GetParallelOperations(matrix1, matrix2, 2, 3), std::invalid_argument);  // NOLINT
}

TEST(Parallel_Operations_SEQ, Throws_On_Invalid_Matrix2_Size) {
  std::vector<int> matrix1 = {1, 2, 3, 4};
  std::vector<int> matrix2 = {1, 2, 3};
  EXPECT_THROW(GetParallelOperations(matrix1, matrix2, 2, 2), std::invalid_argument);  // NOLINT
}
