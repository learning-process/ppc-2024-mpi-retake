// Copyright 2025 Tarakanov Denis
#include "seq/tarakanov_d_integration_the_trapezoid_method/include/ops_seq.hpp"

bool tarakanov_d_integration_the_trapezoid_method_seq::IntegrationTheTrapezoidMethodSequential::PreProcessingImpl() {
  // Init value for input and output
  a = *reinterpret_cast<double*>(task_data->inputs[0]);
  b = *reinterpret_cast<double*>(task_data->inputs[1]);
  h = *reinterpret_cast<double*>(task_data->inputs[2]);
  res = 0.0;

  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_seq::IntegrationTheTrapezoidMethodSequential::ValidationImpl() {
  bool result = task_data->inputs_count[0] == 3 && task_data->outputs_count[0] > 0 && task_data->outputs_count[0] == 1;

  return result;
}

bool tarakanov_d_integration_the_trapezoid_method_seq::IntegrationTheTrapezoidMethodSequential::RunImpl() {
  int n = static_cast<int>((b - a) / h);
  double integral = 0.0;

  for (int i = 0; i < n; ++i) {
    double x0 = a + i * h;
    double x1 = a + (i + 1) * h;
    integral += 0.5 * (func_to_integrate(x0) + func_to_integrate(x1)) * h;
  }

  if (n * h + a < b) {
    double x0 = a + n * h;
    double x1 = b;
    integral += 0.5 * (func_to_integrate(x0) + func_to_integrate(x1)) * (b - n * h);
  }

  res = integral;

  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_seq::IntegrationTheTrapezoidMethodSequential::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = res;
  return true;
}
