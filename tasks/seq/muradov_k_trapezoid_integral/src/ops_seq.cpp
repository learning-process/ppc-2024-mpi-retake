#include "seq/muradov_k_trapezoid_integral/include/ops_seq.hpp"
#include <cmath>

namespace muradov_k_trap_integral_seq {

TrapezoidalIntegral::TrapezoidalIntegral(std::shared_ptr<ppc::core::TaskData> taskData) : Task(taskData) {}

bool TrapezoidalIntegral::PreProcessingImpl() {
  auto input = reinterpret_cast<double*>(task_data->inputs[0]);
  a = input[0];
  b = input[1];
  n = *reinterpret_cast<int*>(task_data->inputs[1]);
  return true;
}

bool TrapezoidalIntegral::ValidationImpl() {
  return (n > 0) && (b > a);
}

bool TrapezoidalIntegral::RunImpl() {
  const double h = (b - a)/n;
  double sum = 0.5*(func(a) + func(b));

  for (int i = 1; i < n; ++i) {
    sum += func(a + i*h);
  }

  result = sum*h;
  return true;
}

bool TrapezoidalIntegral::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result;
  return true;
}

double TrapezoidalIntegral::func(double x) {
  return x*x;
}
}  // namespace muradov_k_trap_integral_seq