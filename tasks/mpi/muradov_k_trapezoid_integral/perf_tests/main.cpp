#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>

#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"

namespace muradov_k_trapezoid_integral_mpi {

// Test #1: measure performance of entire "task run".
TEST(muradov_k_trapezoid_integral_mpi, test_task_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::function<double(double)> f = [](double x) { return x * sin(x); };
  double a = 0.0;
  double b = 10.0;
  int n = 10000000;

  auto start = std::chrono::high_resolution_clock::now();
  double result = GetIntegralTrapezoidalRuleParallel(f, a, b, n);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;

  if (rank == 0) {
    std::cout << "[MPI Task Run] Result: " << result << ", Time: " << elapsed.count() << " seconds\n";
  }
  SUCCEED();
}

// Test #2: measure performance of "pipeline".
TEST(muradov_k_trapezoid_integral_mpi, test_pipeline_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::function<double(double)> f = [](double x) { return exp(x); };
  double a = -6.0;
  double b = 6.0;
  int n = 10000000;

  auto start = std::chrono::high_resolution_clock::now();
  double result = GetIntegralTrapezoidalRuleParallel(f, a, b, n);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;

  if (rank == 0) {
    std::cout << "[MPI Pipeline Run] Result: " << result << ", Time: " << elapsed.count() << " seconds\n";
  }
  SUCCEED();
}

}  // namespace muradov_k_trapezoid_integral_mpi
