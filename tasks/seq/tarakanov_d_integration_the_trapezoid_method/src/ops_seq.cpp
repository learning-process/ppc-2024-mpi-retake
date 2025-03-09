// Copyright 2025 Tarakanov Denis
#include "seq/tarakanov_d_integration_the_trapezoid_method/include/ops_seq.hpp"

bool tarakanov_d_integration_the_trapezoid_method_seq::IntegrationTheTrapezoidMethodSequential::PreProcessingImpl() {
  // Init value for input and output
  a_ = *reinterpret_cast<double*>(task_data->inputs[0]);
  b_ = *reinterpret_cast<double*>(task_data->inputs[1]);
  h_ = *reinterpret_cast<double*>(task_data->inputs[2]);
  res_ = 0.0;

  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_seq::IntegrationTheTrapezoidMethodSequential::ValidationImpl() {
  bool result = task_data->inputs_count[0] == 3 && task_data->outputs_count[0] > 0 && task_data->outputs_count[0] == 1;

  return result;
}

bool tarakanov_d_integration_the_trapezoid_method_seq::IntegrationTheTrapezoidMethodSequential::RunImpl() {
  int n = static_cast<int>((b_ - a_) / h_);
  double integral = 0.0;

  for (int i = 0; i < n; ++i) {
    double x0 = a_ + (i * h_);
    double x1 = a_ + ((i + 1) * h_);
    integral += 0.5 * (FuncToIntegrate(x0) + FuncToIntegrate(x1)) * h_;
  }

  if (n * h_ + a_ < b_) {
    double x0 = a_ + (n * h_);
    double x1 = b_;
    integral += 0.5 * (FuncToIntegrate(x0) + FuncToIntegrate(x1)) * (b_ - n * h_);
  }

  res_ = integral;

  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_seq::IntegrationTheTrapezoidMethodSequential::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = res_;
  return true;
}
