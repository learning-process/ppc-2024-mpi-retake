#pragma once
#include "core/task/include/task.hpp"
#include <memory>


namespace muradov_k_trap_integral_mpi {

class TrapezoidalIntegral : public ppc::core::Task {
public:
  explicit TrapezoidalIntegral(std::shared_ptr<ppc::core::TaskData> task_data);
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

private:
  double a_{}, b_{}, result_{};
  int n_{};
  static double Func(double x);
};
}  // namespace muradov_k_trap_integral_mpi