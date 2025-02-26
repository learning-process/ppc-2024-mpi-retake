#include <gtest/gtest.h>
#include <mpi.h>

#include <iostream>
#include <vector>

#include "mpi/muradov_k_odd_even_batcher_sort/include/ops_mpi.hpp"

namespace mk = muradov_k_odd_even_batcher_sort;

TEST(muradov_k_odd_even_batcher_sort_perf, test_pipeline_run) {
  int procRank, procCount;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
  MPI_Comm_size(MPI_COMM_WORLD, &procCount);
  const int kIterations = 100;
  // Use a large vector (e.g., 256K integers ~1MB if int is 4 bytes)
  const int n = 256 * 1024;
  std::vector<int> original;
  if (procRank == 0) {
    original = mk::random_vector(n);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  for (int i = 0; i < kIterations; i++) {
    std::vector<int> v;
    if (procRank == 0) {
      v = original;  // copy original unsorted vector
    }
    mk::odd_even_batcher_sort(v);
  }
  double end = MPI_Wtime();
  if (procRank == 0) {
    double total_time = end - start;
    double avg_time = total_time / kIterations;
    std::cout << "test_pipeline_run: Average parallel sort time: " << avg_time << " seconds." << std::endl;
  }
  SUCCEED();
}

TEST(muradov_k_odd_even_batcher_sort_perf, test_task_run) {
  int procRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
  // Use a small vector for a single sort task.
  const int n = 1024;
  std::vector<int> v;
  if (procRank == 0) {
    v = mk::random_vector(n);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  mk::odd_even_batcher_sort(v);
  double end = MPI_Wtime();
  if (procRank == 0) {
    double elapsed = end - start;
    std::cout << "test_task_run: Parallel sort round-trip time: " << elapsed << " seconds." << std::endl;
  }
  SUCCEED();
}