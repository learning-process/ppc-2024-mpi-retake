#define OMPI_SKIP_MPICXX
#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "mpi/ersoz_b_horizontal_a_vertical_b/include/ops_mpi.hpp"

TEST(Generation_Matrix, can_generate_square_matrix) {
  auto mat = getRandomMatrix(10, 10);
  ASSERT_EQ(mat.size(), 100);
}

TEST(Generation_Matrix, can_generate_arbitrary_matrix) {
  auto mat = getRandomMatrix(10, 15);
  ASSERT_EQ(mat.size(), 150);
}

TEST(Sequential_Operations_MPI, getSequentialOperations_can_work_with_square_matrix) {
  std::vector<int> matrix1 = getRandomMatrix(10, 10);
  std::vector<int> matrix2 = getRandomMatrix(10, 10);
  auto res = getSequentialOperations(matrix1, matrix2, 10, 10, 10);
  ASSERT_EQ(res.size(), 100);
}

TEST(Sequential_Operations_MPI, getSequentialOperations_can_work_with_arbitrary_matrix) {
  std::vector<int> matrix1 = getRandomMatrix(10, 15);
  std::vector<int> matrix2 = getRandomMatrix(15, 10);
  auto res = getSequentialOperations(matrix1, matrix2, 10, 15, 10);
  ASSERT_EQ(res.size(), 100);
}

TEST(Sequential_Operations_MPI, getSequentialOperations_works_correctly_with_square_matrix) {
  std::vector<int> matrix1 = {1, 1, 1, 1};
  std::vector<int> matrix2 = {2, 2, 2, 2};
  std::vector<int> expected = {4, 4, 4, 4};
  auto res = getSequentialOperations(matrix1, matrix2, 2, 2, 2);
  ASSERT_EQ(res, expected);
}

TEST(Sequential_Operations_MPI, getSequentialOperations_works_correctly_with_arbitrary_matrix) {
  std::vector<int> matrix1 = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> matrix2 = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int> expected = {8, 8, 8, 8, 8, 8};
  auto res = getSequentialOperations(matrix1, matrix2, 2, 4, 3);
  ASSERT_EQ(res, expected);
}

TEST(Parallel_Operations_MPI, getParallelOperations_can_work_with_square_matrix) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> matrix1 = getRandomMatrix(20, 20);
  std::vector<int> matrix2 = getRandomMatrix(20, 20);
  auto res = getParallelOperations(matrix1, matrix2, 20, 20);
  if (rank == 0) {
    ASSERT_EQ(res.size(), 20 * 20);
  }
}

TEST(Parallel_Operations_MPI, getParallelOperations_can_work_with_arbitrary_matrix) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> matrix1 = getRandomMatrix(20, 30);
  std::vector<int> matrix2 = getRandomMatrix(30, 20);
  auto res = getParallelOperations(matrix1, matrix2, 20, 30);
  if (rank == 0) {
    ASSERT_EQ(res.size(), 20 * 20);
  }
}

TEST(Parallel_Operations_MPI, getParallelOperations_works_correctly) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::size_t rows = 20, cols = 20;
  auto matrix1 = getRandomMatrix(rows, cols);
  auto matrix2 = getRandomMatrix(cols, rows);
  auto res_parallel = getParallelOperations(matrix1, matrix2, rows, cols);
  auto res_sequential = getSequentialOperations(matrix1, matrix2, rows, cols, rows);
  if (rank == 0) {
    ASSERT_EQ(res_parallel, res_sequential);
  }
}
