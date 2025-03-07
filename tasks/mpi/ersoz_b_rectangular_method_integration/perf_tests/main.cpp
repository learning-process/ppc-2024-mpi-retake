#define OMPI_SKIP_MPICXX

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>

#include "mpi/ersoz_b_rectangular_method_integration/include/ops_mpi.hpp"

namespace ersoz_b_rectangular_method_integration_mpi {

TEST(ersoz_b_rectangular_method_integration_mpi, test_task_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::function<double(double)> f = [](double x) { return std::cos(x); };
  double a = 0.0;
  double b = 100.0;
  size_t count = 10000000;

  double parallel_result = GetIntegralRectangularMethodParallel(f, a, b, count);

  if (rank == 0) {
    double sequential_result = GetIntegralRectangularMethodSequential(f, a, b, count);
    ASSERT_NEAR(parallel_result, sequential_result, std::numeric_limits<double>::epsilon() * 1000);
  }
}

TEST(ersoz_b_rectangular_method_integration_mpi, test_pipeline_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::function<double(double)> f = [](double x) { return std::cos(x); };
  double a = 0.0;
  double b = 1000.0;
  size_t count = 10000000;

  double parallel_result = GetIntegralRectangularMethodParallel(f, a, b, count);

  if (rank == 0) {
    double sequential_result = GetIntegralRectangularMethodSequential(f, a, b, count);
    ASSERT_NEAR(parallel_result, sequential_result, std::numeric_limits<double>::epsilon() * 1000);
  }
}

}  // namespace ersoz_b_rectangular_method_integration_mpi
