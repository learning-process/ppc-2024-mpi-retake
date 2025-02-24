#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "mpi/ersoz_b_rectangular_method_integration/include/ops_mpi.hpp"

namespace ersoz_b_rectangular_method_integration_mpi {

// Sequential implementation (no changes)
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

// Parallel implementation with MPI (no OpenMP)
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

  // Define the range for this process
  size_t start = rank * part;
  size_t end = (rank == process_count - 1) ? count : start + part;

  // Compute local result
  for (size_t i = start; i < end; ++i) {
    local_result += integrable_function(a + i * delta);
  }
  local_result *= delta;

  // MPI reduction to combine results from all processes
  double result = 0.0;
  MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return result;
}

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_FROM_0_TO_1) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel([](double x) { return std::cos(x); }, 0, 1, 10000);
  if (rank == 0) {
    double sequential_result =
        GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 0, 1, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 1000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_FROM_5_TO_0) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel([](double x) { return std::cos(x); }, 5, 0, 10000);
  if (rank == 0) {
    double sequential_result =
        GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 5, 0, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 1000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_FROM_0_TO_100) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel([](double x) { return std::cos(x); }, 0, 100, 10000);
  if (rank == 0) {
    double sequential_result =
        GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 0, 100, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 1000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_FROM_0_TO_709) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel([](double x) { return std::cos(x); }, 0, 709, 10000);
  if (rank == 0) {
    double sequential_result =
        GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 0, 709, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 10000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_WITH_LOW_RANGE) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel([](double x) { return std::cos(x); }, 1, 1.01, 10000);
  if (rank == 0) {
    double sequential_result =
        GetIntegralRectangularMethodSequential([](double x) { return std::cos(x); }, 1, 1.01, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 1000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, EXCEPTION_ON_ZERO_COUNT) {
  EXPECT_THROW(GetIntegralRectangularMethodParallel([](double x) { return std::cos(x); }, 5, 0, 0), std::runtime_error);
}

}  // namespace ersoz_b_rectangular_method_integration_mpi