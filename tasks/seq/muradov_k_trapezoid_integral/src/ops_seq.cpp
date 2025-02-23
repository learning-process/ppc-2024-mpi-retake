#include "seq/muradov_k_trapezoid_integral/include/ops_seq.hpp"

#include <cmath>
#include <memory>
#include <utility>

#include "core/task/include/task.hpp"

namespace muradov_k_trap_integral_seq {

TrapezoidalIntegral::TrapezoidalIntegral(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

bool TrapezoidalIntegral::PreProcessingImpl() {
  auto* input = reinterpret_cast<double*>(task_data->inputs[0]);
  a_ = input[0];
  b_ = input[1];
  n_ = *reinterpret_cast<int*>(task_data->inputs[1]);
  return true;
}

bool TrapezoidalIntegral::ValidationImpl() { return (n_ > 0) && (b_ > a_); }

bool TrapezoidalIntegral::RunImpl() {
  const double h = (b_ - a_) / n_;
  double sum = 0.5 * (Func(a_) + Func(b_));

  for (int i = 1; i < n_; ++i) {
    sum += Func(a_ + (static_cast<double>(i) * h));
  }

  result_ = sum * h;
  return true;
}

bool TrapezoidalIntegral::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  return true;
}

double TrapezoidalIntegral::Func(double x) { return x * x; }
}  // namespace muradov_k_trap_integral_seq