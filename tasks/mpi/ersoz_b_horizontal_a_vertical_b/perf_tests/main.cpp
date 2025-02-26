#include <cstddef>
#define OMPI_SKIP_MPICXX
#include <gtest/gtest.h>
#include <mpi.h>

#include <iostream>

#include "mpi/ersoz_b_horizontal_a_vertical_b/include/ops_mpi.hpp"

TEST(ersoz_b_horizontal_a_vertical_b_mpi, test_pipeline_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::size_t rows = 200;
  std::size_t cols = 150;
  auto matrix1 = GetRandomMatrix(rows, cols);
  auto matrix2 = GetRandomMatrix(cols, rows);
  constexpr int kIterations = 100;
  double total_time = 0.0;
  for (int i = 0; i < kIterations; ++i) {
    double start = MPI_Wtime();
    auto res = GetParallelOperations(matrix1, matrix2, rows, cols);
    double end = MPI_Wtime();
    total_time += (end - start);
  }
  if (rank == 0) {
    double avg_time = total_time / kIterations;
    std::cout << "test_pipeline_run - Average Parallel MPI Multiplication Time: " << avg_time << " seconds\n";
  }
  SUCCEED();
}

TEST(ersoz_b_horizontal_a_vertical_b_mpi, test_task_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::size_t rows = 200;
  std::size_t cols = 150;
  std::size_t b_cols = rows;  // Result dimensions: rows x rows.
  auto matrix1 = GetRandomMatrix(rows, cols);
  auto matrix2 = GetRandomMatrix(cols, rows);

  double seq_time = 0.0;
  \n double par_time = 0.0;
  std::vector<int> res_seq;
  std::vector<int> res_par;
  if (rank == 0) {
    double start_seq = MPI_Wtime();
    res_seq = GetSequentialOperations(matrix1, matrix2, rows, cols, b_cols);
    double end_seq = MPI_Wtime();
    seq_time = end_seq - start_seq;
  }
  double start_par = MPI_Wtime();
  res_par = GetParallelOperations(matrix1, matrix2, rows, cols);
  double end_par = MPI_Wtime();
  par_time = end_par - start_par;

  if (rank == 0) {
    std::cout << "test_task_run - Sequential Time: " << seq_time << " seconds, Parallel Time: " << par_time
              << " seconds\n";
    ASSERT_EQ(res_seq, res_par);
  }
  SUCCEED();
}
