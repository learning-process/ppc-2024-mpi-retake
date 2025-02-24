#include <omp.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>

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

}  // namespace ersoz_b_rectangular_method_integration_mpi