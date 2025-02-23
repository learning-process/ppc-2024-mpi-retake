#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

#include "core/task/include/task.hpp"

namespace muradov_k_trap_integral_mpi {

TrapezoidalIntegral::TrapezoidalIntegral(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

bool TrapezoidalIntegral::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    auto* input = reinterpret_cast<double*>(task_data->inputs[0]);
    a_ = input[0];
    b_ = input[1];
    n_ = *reinterpret_cast<int*>(task_data->inputs[1]);
  }

  MPI_Bcast(&a_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool TrapezoidalIntegral::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return (rank != 0) || ((n_ > 0) && (b_ > a_));
}

bool TrapezoidalIntegral::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const double h = (b_ - a_) / n_;
  const int local_n = (n_ / size) + ((rank < (n_ % size)) ? 1 : 0);
  const double local_a = a_ + h * ((rank * (static_cast<double>(n_) / size)) + std::min(rank, n_ % size));
  const double local_b = local_a + (h * local_n);

  double local_sum = 0.5 * (Func(local_a) + Func(local_b));
  for (int i = 1; i < local_n; ++i) {
    local_sum += Func(local_a + (static_cast<double>(i) * h));
  }
  local_sum *= h;

  MPI_Reduce((rank == 0) ? MPI_IN_PLACE : &local_sum, &result_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return true;
}

bool TrapezoidalIntegral::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  }
  return true;
}

double TrapezoidalIntegral::Func(double x) { return x * x; }
}  // namespace muradov_k_trap_integral_mpi