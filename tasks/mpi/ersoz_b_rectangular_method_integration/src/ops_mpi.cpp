#include "mpi/ersoz_b_rectangular_method_integration/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>  // size_t için
#include <functional>
#include <stdexcept>

namespace ersoz_b_rectangular_method_integration_mpi {

double GetIntegralRectangularMethodSequential(const std::function<double(double)>& integrable_function, double a,
                                              double b, size_t count) {
  if (count == 0) {
    throw std::runtime_error("Zero rectangles count");
  }
  double result = 0.0;
  double delta = (b - a) / static_cast<double>(count);  // Daraltma dönüşümünü önle
  for (size_t i = 0; i < count; i++) {
    double x = a + (i * delta);  // Parantez ekleyin
    result += integrable_function(x);
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

  double delta = (b - a) / static_cast<double>(count);  // Daraltma dönüşümünü önle
  size_t part = count / static_cast<size_t>(process_count);
  double local_result = 0.0;

  // Define the range for this process
  size_t start = rank * part;
  size_t end = (rank == process_count - 1) ? count : start + part;

  // Compute local result
  for (size_t i = start; i < end; ++i) {
    double x = a + (i * delta);  // Parantez ekleyin
    local_result += integrable_function(x);
  }
  local_result *= delta;

  // MPI reduction to combine results from all processes
  double result = 0.0;
  MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return result;
}

}  // namespace ersoz_b_rectangular_method_integration_mpi