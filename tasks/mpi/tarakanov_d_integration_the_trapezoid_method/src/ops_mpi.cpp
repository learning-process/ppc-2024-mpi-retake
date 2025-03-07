// Copyright 2025 Tarakanov Denis
#include "mpi/tarakanov_d_integration_the_trapezoid_method/include/ops_mpi.hpp"

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <cstdint>
#include <functional>
#include <vector>

bool tarakanov_d_integration_the_trapezoid_method_mpi::IntegrationTheTrapezoidMethodMPI::PreProcessingImpl() {
  // Init value for input and output
  if (world_.rank() == 0) {
    a_ = *reinterpret_cast<double*>(task_data->inputs[0]);
    b_ = *reinterpret_cast<double*>(task_data->inputs[1]);
    h_ = *reinterpret_cast<double*>(task_data->inputs[2]);
    res_ = 0.0;
  }

  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::IntegrationTheTrapezoidMethodMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    bool result =
        task_data->inputs_count[0] == 3 && task_data->outputs_count[0] > 0 && task_data->outputs_count[0] == 1;
    return result;
  }

  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::IntegrationTheTrapezoidMethodMPI::RunImpl() {
  boost::mpi::broadcast(world_, a_, 0);
  boost::mpi::broadcast(world_, b_, 0);
  boost::mpi::broadcast(world_, h_, 0);

  uint32_t rank = world_.rank();
  uint32_t size = world_.size();

  auto parts_count = static_cast<uint32_t>((b_ - a_) / h_);
  uint32_t local_parts_count = parts_count / size;
  uint32_t start = local_parts_count * rank;
  uint32_t end = (rank == size - 1) ? parts_count : start + local_parts_count;

  double local_res = 0.0;
  for (uint32_t i = start; i < end; ++i) {
    double x0 = a_ + (i * h_);
    double x1 = x0 + h_ > b_ ? b_ : x0 + h_;
    local_res += 0.5 * (FuncToIntegrate(x0) + FuncToIntegrate(x1)) * (x1 - x0);
  }

  boost::mpi::reduce(world_, local_res, res_, std::plus<>(), 0);

  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_mpi::IntegrationTheTrapezoidMethodMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    *reinterpret_cast<double*>(task_data->outputs[0]) = res_;
  }
  return true;
}
