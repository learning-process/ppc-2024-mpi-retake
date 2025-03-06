#define OMPI_SKIP_MPICXX
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "mpi/ersoz_b_horizontal_a_vertical_b/include/ops_mpi.hpp"

TEST(ersoz_b_horizontal_a_vertical_b_mpi, test_pipeline_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::size_t rows = 200;
  std::size_t cols = 150;
  auto matrix1 = GetRandomMatrix(rows, cols);
  auto matrix2 = GetRandomMatrix(cols, rows);
  constexpr int kIterations = 100;
  std::vector<int> res;
  for (int i = 0; i < kIterations; ++i) {
    res = GetParallelOperations(matrix1, matrix2, rows, cols);
  }
  if (rank == 0) {
    auto expected = GetSequentialOperations(matrix1, matrix2, rows, cols, rows);
    ASSERT_EQ(expected, res);
  }
}

TEST(ersoz_b_horizontal_a_vertical_b_mpi, test_task_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::size_t rows = 200;
  std::size_t cols = 150;
  std::size_t b_cols = rows;
  auto matrix1 = GetRandomMatrix(rows, cols);
  auto matrix2 = GetRandomMatrix(cols, rows);
  std::vector<int> res_seq;
  std::vector<int> res_par;
  if (rank == 0) {
    res_seq = GetSequentialOperations(matrix1, matrix2, rows, cols, b_cols);
  }
  res_par = GetParallelOperations(matrix1, matrix2, rows, cols);
  if (rank == 0) {
    ASSERT_EQ(res_seq, res_par);
  }
  SUCCEED();
}
