// Copyright 2025 Tarakanov Denis
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace tarakanov_d_integration_the_trapezoid_method_mpi {

class IntegrationTheTrapezoidMethodMPI : public ppc::core::Task {
 public:
  explicit IntegrationTheTrapezoidMethodMPI(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double a{}, b{}, h{}, res{};
  static double func_to_integrate(double x) { return x / 2; };
  boost::mpi::communicator world;
};

}  // namespace tarakanov_d_integration_the_trapezoid_method_mpi