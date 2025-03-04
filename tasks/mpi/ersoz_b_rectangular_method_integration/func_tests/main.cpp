#define OMPI_SKIP_MPICXX

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "mpi/ersoz_b_rectangular_method_integration/include/ops_mpi.hpp"

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

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_SIN_FROM_0_TO_PI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel([](double x) { return std::sin(x); }, 0, M_PI, 10000);
  if (rank == 0) {
    double sequential_result =
        GetIntegralRectangularMethodSequential([](double x) { return std::sin(x); }, 0, M_PI, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 1000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_X_SQUARE_FROM_0_TO_1) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel([](double x) { return x * x; }, 0, 1, 10000);
  if (rank == 0) {
    double sequential_result = GetIntegralRectangularMethodSequential([](double x) { return x * x; }, 0, 1, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 1000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, INTEGRAL_EXP_FROM_0_TO_1) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double parallel_result = GetIntegralRectangularMethodParallel([](double x) { return std::exp(x); }, 0, 1, 10000);
  if (rank == 0) {
    double sequential_result =
        GetIntegralRectangularMethodSequential([](double x) { return std::exp(x); }, 0, 1, 10000);
    ASSERT_LT(std::fabs(parallel_result - sequential_result), std::numeric_limits<double>::epsilon() * 1000);
  }
}

}  // namespace ersoz_b_rectangular_method_integration_mpi
