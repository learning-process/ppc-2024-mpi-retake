#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>

#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"

namespace muradov_k_trapezoid_integral_mpi {

TEST(muradov_k_trapezoid_integral_mpi, test_task_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::function<double(double)> f = [](double x) { return x * sin(x); };
  double a = 0.0;
  double b = 10.0;
  int n = 10000000;

  double result = GetIntegralTrapezoidalRuleParallel(f, a, b, n);

  if (rank == 0) {
    double expected_value = 9.09304866;
    double tolerance = 1e-4;
    ASSERT_NEAR(result, expected_value, tolerance);
  }
}

TEST(muradov_k_trapezoid_integral_mpi, test_pipeline_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::function<double(double)> f = [](double x) { return exp(x); };
  double a = -6.0;
  double b = 6.0;
  int n = 10000000;

  double result = GetIntegralTrapezoidalRuleParallel(f, a, b, n);

  if (rank == 0) {
    double expected_value = exp(6) - exp(-6);
    double tolerance = 1e-4;
    ASSERT_NEAR(result, expected_value, tolerance);
  }
}

}  // namespace muradov_k_trapezoid_integral_mpi
