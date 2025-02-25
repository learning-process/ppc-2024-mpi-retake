#define OMPI_SKIP_MPICXX

#include "mpi/ersoz_b_rectangular_method_integration/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>

namespace {
// Helper function to compute the integral sequentially over a given range.
double integrateSequential(const std::function<double(double)>& f, double a, double b, size_t count) {
  if (count == 0) {
    throw std::runtime_error("Zero rectangles count");
  }
  double result = 0.0;
  double delta = (b - a) / static_cast<double>(count);
  for (size_t i = 0; i < count; ++i) {
    result += f(a + i * delta);
  }
  return result * delta;
}
}  // namespace

namespace ersoz_b_rectangular_method_integration_mpi {

double GetIntegralRectangularMethodSequential(const std::function<double(double)>& f, double a, double b,
                                              size_t count) {
  return integrateSequential(f, a, b, count);
}

double GetIntegralRectangularMethodParallel(const std::function<double(double)>& f, double a, double b, size_t count) {
  if (count == 0) {
    throw std::runtime_error("Zero rectangles count");
  }

  int rank = 0, process_count = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double delta = (b - a) / static_cast<double>(count);
  size_t part = count / static_cast<size_t>(process_count);
  size_t start = rank * part;
  size_t end = (rank == process_count - 1) ? count : start + part;
  double local_result = 0.0;

  for (size_t i = start; i < end; ++i) {
    local_result += f(a + i * delta);
  }
  local_result *= delta;

  double global_result = 0.0;
  MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return global_result;
}

}  // namespace ersoz_b_rectangular_method_integration_mpi
