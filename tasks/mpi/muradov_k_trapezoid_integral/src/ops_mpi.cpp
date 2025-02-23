#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"
#include <mpi.h>
#include <cmath>
#include <algorithm>

namespace muradov_k_trap_integral_mpi {

TrapezoidalIntegral::TrapezoidalIntegral(std::shared_ptr<ppc::core::TaskData> taskData) : Task(taskData) {}

bool TrapezoidalIntegral::PreProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    auto input = reinterpret_cast<double*>(taskData->inputs[0]);
    a = input[0];
    b = input[1];
    n = *reinterpret_cast<int*>(taskData->inputs[1]);
  }

  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool TrapezoidalIntegral::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return (rank != 0) || ((n > 0) && (b > a));
}

bool TrapezoidalIntegral::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const double h = (b - a)/n;
  const int local_n = n/size + ((rank < n%size) ? 1 : 0);
  const double local_a = a + h*(rank*(n/size) + std::min(rank, n%size));
  const double local_b = local_a + h*local_n;

  double local_sum = 0.5*(func(local_a) + func(local_b));
  for (int i = 1; i < local_n; ++i) {
    local_sum += func(local_a + i*h);
  }
  local_sum *= h;

  MPI_Reduce((rank == 0) ? MPI_IN_PLACE : &local_sum, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return true;
}

bool TrapezoidalIntegral::PostProcessingImpl() {
  if (MPI_Comm_rank(MPI_COMM_WORLD, nullptr) == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = result;
  }
  return true;
}

double TrapezoidalIntegral::func(double x) {
  return x*x;
}
}  // namespace muradov_k_trap_integral_mpi