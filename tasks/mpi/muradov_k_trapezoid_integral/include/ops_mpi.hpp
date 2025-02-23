#pragma once
#include <memory>
#include "core/task/include/task.hpp"

namespace muradov_k_trap_integral_mpi {

class TrapezoidalIntegral : public ppc::core::Task {
public:
  explicit TrapezoidalIntegral(std::shared_ptr<ppc::core::TaskData> taskData);
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

private:
  double a{}, b{}, result{};
  int n{};
  static double func(double x);
};
}  // namespace muradov_k_trap_integral_mpi