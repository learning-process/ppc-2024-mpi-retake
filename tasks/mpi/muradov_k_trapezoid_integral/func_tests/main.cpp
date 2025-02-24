#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>

#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"

namespace muradov_k_trapezoid_integral_mpi {

TEST(muradov_k_trapezoid_integral_mpi, SquareFunction) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::function<double(double)> f = [](double x) { return x * x; };
  double a = 5.0;
  double b = 10.0;
  int n = 100;

  double global_sum = GetIntegralTrapezoidalRuleParallel(f, a, b, n);

  if (rank == 0) {
    double reference_sum = 0.0;
    double h = (b - a) / static_cast<double>(n);
    for (int i = 0; i < n; i++) {
      double x_i = a + i * h;
      double x_next = a + ((i + 1) * h);
      reference_sum += (f(x_i) + f(x_next)) * 0.5 * h;
    }
    ASSERT_NEAR(reference_sum, global_sum, 1e-6);
  }
}

TEST(muradov_k_trapezoid_integral_mpi, CubeFunction) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::function<double(double)> f = [](double x) { return x * x * x; };
  double a = 0.0;
  double b = 6.0;
  int n = 100;

  double global_sum = GetIntegralTrapezoidalRuleParallel(f, a, b, n);

  if (rank == 0) {
    double reference_sum = 0.0;
    double h = (b - a) / static_cast<double>(n);
    for (int i = 0; i < n; i++) {
      double x_i = a + i * h;
      double x_next = a + ((i + 1) * h);
      reference_sum += (f(x_i) + f(x_next)) * 0.5 * h;
    }
    ASSERT_NEAR(reference_sum, global_sum, 1e-6);
  }
}

}  // namespace muradov_k_trapezoid_integral_mpi
