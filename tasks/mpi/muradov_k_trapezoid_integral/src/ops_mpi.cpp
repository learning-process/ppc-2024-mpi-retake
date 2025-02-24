#include <functional>  // for std::function
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>

#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"

namespace muradov_k_trapezoid_integral_mpi {

double GetIntegralTrapezoidalRuleParallel(const std::function<double(double)>& f, double a, double b, int n) {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (n <= 0) {
    return 0.0;
  }

  double h = (b - a) / static_cast<double>(n);
  double local_sum = 0.0;

  // Each process computes a portion of the integral.
  for (int i = rank; i < n; i += size) {
    double x_i = a + (i * h);
    double x_next = a + ((i + 1) * h);
    local_sum += (f(x_i) + f(x_next)) * 0.5 * h;
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return global_sum;
}

}  // namespace muradov_k_trapezoid_integral_mpi
