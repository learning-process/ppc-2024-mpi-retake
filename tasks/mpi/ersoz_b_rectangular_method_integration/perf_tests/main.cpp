#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>

#include "mpi/ersoz_b_rectangular_method_integration/include/ops_mpi.hpp"

namespace ersoz_b_rectangular_method_integration_mpi {

TEST(ersoz_b_rectangular_method_integration_mpi, test_task_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::function<double(double)> f = [](double x) { return cos(x); };
  double a = 0.0;
  double b = 100.0;
  size_t count = 10000000;

  auto start = std::chrono::high_resolution_clock::now();
  double result = GetIntegralRectangularMethodParallel(f, a, b, count);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  if (rank == 0) {
    std::cout << "[MPI Task Run] Result: " << result << ", Time: " << elapsed.count() << " seconds\n";
  }
  SUCCEED();
}

TEST(ersoz_b_rectangular_method_integration_mpi, test_pipeline_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::function<double(double)> f = [](double x) { return cos(x); };
  double a = 0.0;
  double b = 1000.0;
  size_t count = 10000000;

  auto start = std::chrono::high_resolution_clock::now();
  double result = GetIntegralRectangularMethodParallel(f, a, b, count);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  if (rank == 0) {
    std::cout << "[MPI Pipeline Run] Result: " << result << ", Time: " << elapsed.count() << " seconds\n";
  }
  SUCCEED();
}

}  // namespace ersoz_b_rectangular_method_integration_mpi