#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

#include "mpi/ersoz_b_rectangular_method_integration/include/ops_mpi.hpp"

namespace ersoz_b_rectangular_method_integration_mpi {

double GetIntegralRectangularMethodSequential(const std::function<double(double)>& integrable_function, double a,
                                              double b, size_t count) {
  if (count == 0) {
    throw std::runtime_error("Zero rectangles count");
  }
  double result = 0.0;
  double delta = (b - a) / static_cast<double>(count);
  for (size_t i = 0; i < count; i++) {
    result += integrable_function(a + (i * delta));
  }
  result *= delta;
  return result;
}

double GetIntegralRectangularMethodParallel(const std::function<double(double)>& integrable_function, double a,
                                            double b, size_t count) {
  if (count == 0) {
    throw std::runtime_error("Zero rectangles count");
  }

  int rank = 0;
  int process_count = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double delta = (b - a) / static_cast<double>(count);
  size_t part = count / static_cast<size_t>(process_count);
  double local_result = 0.0;

  size_t start = rank * part;
  size_t end = (rank == process_count - 1) ? count : start + part;

#pragma omp parallel for reduction(+ : local_result)
  for (size_t i = start; i < end; ++i) {
    local_result += integrable_function(a + i * delta);
  }
  local_result *= delta;

  double result = 0.0;
  MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return result;
}

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