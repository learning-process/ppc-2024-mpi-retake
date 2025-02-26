#include <gtest/gtest.h>

#include <chrono>

#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

TEST(DiningPhilosophersPerf, test_pipeline_run_mpi) {
  int argc = 0;
  char** argv = nullptr;
  MPI_Init(&argc, &argv);

  dining_philosophers::DiningPhilosophersMPI philosophers(100);
  auto start = std::chrono::high_resolution_clock::now();
  philosophers.Run();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Pipeline execution time: " << duration.count() << "s\n";

  MPI_Finalize();
}

TEST(DiningPhilosophersPerf, test_task_run_mpi) {
  int argc = 0;
  char** argv = nullptr;
  MPI_Init(&argc, &argv);

  dining_philosophers::DiningPhilosophersMPI philosophers(100);
  auto start = std::chrono::high_resolution_clock::now();
  philosophers.Run();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Task execution time: " << duration.count() << "s\n";

  MPI_Finalize();
}
