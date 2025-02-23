// Copyright 2025 Tarakanov Denis
#include "mpi/tarakanov_d_integration_the_trapezoid_method/include/ops_mpi.hpp"

#include <vector>

bool tarakanov_d_integration_the_trapezoid_method_mpi::IntegrationTheTrapezoidMethodMPI::PreProcessingImpl() {
  // Init value for input and output
  if (world.rank() == 0) {
    a = *reinterpret_cast<double*>(task_data->inputs[0]);
    b = *reinterpret_cast<double*>(task_data->inputs[1]);
    h = *reinterpret_cast<double*>(task_data->inputs[2]);
    res = 0.0;
  }

  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::IntegrationTheTrapezoidMethodMPI::ValidationImpl() {
  if (world.rank() == 0) {
    bool result =
        task_data->inputs_count[0] == 3 && task_data->outputs_count[0] > 0 && task_data->outputs_count[0] == 1;
    return result;
  }

  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::IntegrationTheTrapezoidMethodMPI::RunImpl() {
  boost::mpi::broadcast(world, a, 0);
  boost::mpi::broadcast(world, b, 0);
  boost::mpi::broadcast(world, h, 0);

  uint32_t rank = world.rank();
  uint32_t size = world.size();

  uint32_t partsCount = static_cast<int>((b - a) / h);
  uint32_t localPartsCount = partsCount / size;
  uint32_t start = localPartsCount * rank;
  uint32_t end = (rank == size - 1) ? partsCount : start + localPartsCount;

  double local_res = 0.0;
  for (uint32_t i = start; i < end; ++i) {
    double x0 = a + i * h;
    double x1 = (rank == size - 1) ? b : a + (i + 1) * h;
    local_res += 0.5 * (func_to_integrate(x0) + func_to_integrate(x1)) * (x1 - x0);
  }

  boost::mpi::reduce(world, local_res, res, std::plus<double>(), 0);

  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::IntegrationTheTrapezoidMethodMPI::PostProcessingImpl() {
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(task_data->outputs[0]) = res;
  }
  return true;
}
