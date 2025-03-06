#define OMPI_SKIP_MPICXX
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "mpi/ersoz_b_horizontal_a_vertical_b/include/ops_mpi.hpp"

// Test: Generation of matrices
TEST(Generation_Matrix, can_generate_square_matrix) {
  auto mat = GetRandomMatrix(10, 10);
  ASSERT_EQ(mat.size(), 100U);
}

TEST(Generation_Matrix, can_generate_arbitrary_matrix) {
  auto mat = GetRandomMatrix(10, 15);
  ASSERT_EQ(mat.size(), 150U);
}

// Tests for sequential operations.
TEST(Sequential_Operations_MPI, GetSequentialOperations_can_work_with_square_matrix) {
  std::vector<int> matrix1 = GetRandomMatrix(10, 10);
  std::vector<int> matrix2 = GetRandomMatrix(10, 10);
  auto res = GetSequentialOperations(matrix1, matrix2, 10, 10, 10);
  ASSERT_EQ(res.size(), 100U);
}

TEST(Sequential_Operations_MPI, GetSequentialOperations_can_work_with_arbitrary_matrix) {
  std::vector<int> matrix1 = GetRandomMatrix(10, 15);
  std::vector<int> matrix2 = GetRandomMatrix(15, 10);
  auto res = GetSequentialOperations(matrix1, matrix2, 10, 15, 10);
  ASSERT_EQ(res.size(), 100U);
}

TEST(Sequential_Operations_MPI, GetSequentialOperations_works_correctly_with_square_matrix) {
  std::vector<int> matrix1 = {1, 1, 1, 1};
  std::vector<int> matrix2 = {2, 2, 2, 2};
  std::vector<int> expected = {4, 4, 4, 4};
  auto res = GetSequentialOperations(matrix1, matrix2, 2, 2, 2);
  ASSERT_EQ(expected, res);
}

TEST(Sequential_Operations_MPI, GetSequentialOperations_works_correctly_with_arbitrary_matrix) {
  std::vector<int> matrix1 = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> matrix2 = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int> expected = {8, 8, 8, 8, 8, 8};
  auto res = GetSequentialOperations(matrix1, matrix2, 2, 4, 3);
  ASSERT_EQ(expected, res);
}

// Tests for parallel operations.
TEST(Parallel_Operations_MPI, GetParallelOperations_can_work_with_square_matrix) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> matrix1 = GetRandomMatrix(20, 20);
  std::vector<int> matrix2 = GetRandomMatrix(20, 20);
  auto res = GetParallelOperations(matrix1, matrix2, 20, 20);
  if (rank == 0) {
    ASSERT_EQ(res.size(), 20U * 20U);
  }
}

TEST(Parallel_Operations_MPI, GetParallelOperations_can_work_with_arbitrary_matrix) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> matrix1 = GetRandomMatrix(20, 30);
  std::vector<int> matrix2 = GetRandomMatrix(30, 20);
  auto res = GetParallelOperations(matrix1, matrix2, 20, 30);
  if (rank == 0) {
    ASSERT_EQ(res.size(), 20U * 20U);
  }
}

TEST(Parallel_Operations_MPI, GetParallelOperations_works_correctly) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const std::size_t rows = 20;
  const std::size_t cols = 20;
  auto matrix1 = GetRandomMatrix(rows, cols);
  auto matrix2 = GetRandomMatrix(cols, rows);
  auto res_parallel = GetParallelOperations(matrix1, matrix2, rows, cols);
  auto res_sequential = GetSequentialOperations(matrix1, matrix2, rows, cols, rows);
  if (rank == 0) {
    ASSERT_EQ(res_sequential, res_parallel);
  }
}
