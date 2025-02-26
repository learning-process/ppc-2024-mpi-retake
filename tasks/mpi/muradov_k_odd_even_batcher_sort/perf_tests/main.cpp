#define OMPI_SKIP_MPICXX

#include <gtest/gtest.h>
#include <mpi.h>

#include <iostream>
#include <vector>

#include "mpi/muradov_k_odd_even_batcher_sort/include/ops_mpi.hpp"

namespace mk = muradov_k_odd_even_batcher_sort;

TEST(muradov_k_odd_even_batcher_sort_mpi, test_pipeline_run) {
  int proc_rank = 0;
  int proc_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  const int k_iterations = 100;
  // Use a large vector (e.g., 256K integers ~1MB if int is 4 bytes)
  const int n = 256 * 1024;
  std::vector<int> original;
  if (proc_rank == 0) {
    original = mk::RandomVector(n);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  for (int i = 0; i < k_iterations; i++) {
    std::vector<int> v;
    if (proc_rank == 0) {
      v = original;  // copy original unsorted vector
    }
    mk::OddEvenBatcherSort(v);
  }
  double end = MPI_Wtime();
  if (proc_rank == 0) {
    double total_time = end - start;
    double avg_time = total_time / k_iterations;
    std::cout << "test_pipeline_run: Average parallel sort time: " << avg_time << " seconds.\n";
  }
  SUCCEED();
}

TEST(muradov_k_odd_even_batcher_sort_mpi, test_task_run) {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  // Use a small vector for a single sort task.
  const int n = 1024;
  std::vector<int> v;
  if (proc_rank == 0) {
    v = mk::RandomVector(n);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  mk::OddEvenBatcherSort(v);
  double end = MPI_Wtime();
  if (proc_rank == 0) {
    double elapsed = end - start;
    std::cout << "test_task_run: Parallel sort round-trip time: " << elapsed << " seconds.\n";
  }
  SUCCEED();
}
