#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <limits>
#include <stdexcept>

// Include the header file
#include "mpi/ersoz_b_rectangular_method_integration/include/ops_mpi.hpp"

namespace ersoz_b_rectangular_method_integration_mpi {

// Function definitions matching the header file's declarations
double GetIntegralRectangularMethodSequential(const std::function<double(double)>& integrable_function, double a,
                                              double b, size_t count) {
  if (count <= 0) {
    throw std::runtime_error("Number of intervals must be greater than zero.");
  }

  double h = (b - a) / count;
  double result = 0.0;

  for (size_t i = 0; i < count; ++i) {
    double x = a + (i + 0.5) * h;
    result += integrable_function(x);
  }

  result *= h;
  return result;
}

double GetIntegralRectangularMethodParallel(const std::function<double(double)>& integrable_function, double a,
                                            double b, size_t count) {
  double result = 0.0;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (count <= 0) {
    throw std::runtime_error("Number of intervals must be greater than zero.");
  }

  double h = (b - a) / count;
  double local_result = 0.0;
  size_t local_n = count / size;
  size_t local_start = rank * local_n;
  size_t local_end = (rank == size - 1) ? count : local_start + local_n;

  for (size_t i = local_start; i < local_end; ++i) {
    double x = a + (i + 0.5) * h;
    local_result += integrable_function(x);
  }

  local_result *= h;

  MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return result;
}

}  // namespace ersoz_b_rectangular_method_integration_mpi

// Test cases
namespace ersoz_b_rectangular_method_integration_mpi {

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
