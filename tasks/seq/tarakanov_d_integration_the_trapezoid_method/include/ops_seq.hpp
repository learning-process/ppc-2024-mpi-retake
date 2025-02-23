// Copyright 2025 Tarakanov Denis
#pragma once

#include "core/task/include/task.hpp"

namespace tarakanov_d_integration_the_trapezoid_method_seq {

class IntegrationTheTrapezoidMethodSequential : public ppc::core::Task {
 public:
  explicit IntegrationTheTrapezoidMethodSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double a{}, b{}, h{}, res{};
  static double func_to_integrate(double x) { return x / 2; };
};

}  // namespace tarakanov_d_integration_the_trapezoid_method_seq