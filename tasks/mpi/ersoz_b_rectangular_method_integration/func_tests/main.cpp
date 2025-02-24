#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "mpi/ersoz_b_rectangular_method_integration/include/ops_mpi.hpp"

// Mock or actual implementation of the integration functions
double GetIntegralRectangularMethodParallel(double (*func)(double), double a, double b, int n) {
  // Placeholder for parallel implementation
  double result = 0.0;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (n <= 0) {
    throw std::runtime_error("Number of intervals must be greater than zero.");
  }

  double h = (b - a) / n;
  double local_result = 0.0;
  int local_n = n / size;
  int local_start = rank * local_n;
  int local_end = (rank == size - 1) ? n : local_start + local_n;

  for (int i = local_start; i < local_end; ++i) {
    double x = a + (i + 0.5) * h;
    local_result += func(x);
  }

  local_result *= h;

  MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return result;
}

double GetIntegralRectangularMethodSequential(double (*func)(double), double a, double b, int n) {
  // Placeholder for sequential implementation
  if (n <= 0) {
    throw std::runtime_error("Number of intervals must be greater than zero.");
  }

  double h = (b - a) / n;
  double result = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = a + (i + 0.5) * h;
    result += func(x);
  }

  result *= h;
  return result;
}

namespace ersoz_b_rectangular_method_integration_mpi {

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_FROM_0_TO_1) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel(static_cast<double (*)(double)>(&cos), 0, 1, 10000);
  if (rank == 0) {
    double sequential_result =
        GetIntegralRectangularMethodSequential(static_cast<double (*)(double)>(&cos), 0, 1, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 1000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_FROM_5_TO_0) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel(static_cast<double (*)(double)>(&cos), 5, 0, 10000);
  if (rank == 0) {
    double sequential_result =
        GetIntegralRectangularMethodSequential(static_cast<double (*)(double)>(&cos), 5, 0, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 1000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_FROM_0_TO_100) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel(static_cast<double (*)(double)>(&cos), 0, 100, 10000);
  if (rank == 0) {
    double sequential_result =
        GetIntegralRectangularMethodSequential(static_cast<double (*)(double)>(&cos), 0, 100, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 1000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_FROM_0_TO_709) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel(static_cast<double (*)(double)>(&cos), 0, 709, 10000);
  if (rank == 0) {
    double sequential_result =
        GetIntegralRectangularMethodSequential(static_cast<double (*)(double)>(&cos), 0, 709, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 10000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_WITH_LOW_RANGE) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel(static_cast<double (*)(double)>(&cos), 1, 1.01, 10000);
  if (rank == 0) {
    double sequential_result =
        GetIntegralRectangularMethodSequential(static_cast<double (*)(double)>(&cos), 1, 1.01, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 1000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, EXCEPTION_ON_ZERO_COUNT) {
  EXPECT_THROW(GetIntegralRectangularMethodParallel(static_cast<double (*)(double)>(&cos), 5, 0, 0),
               std::runtime_error);
}

}  // namespace ersoz_b_rectangular_method_integration_mpi
