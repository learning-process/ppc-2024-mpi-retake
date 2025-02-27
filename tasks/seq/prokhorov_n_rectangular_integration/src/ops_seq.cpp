#include "seq/prokhorov_n_rectangular_integration/include/ops_seq.hpp"

#include <cmath>
#include <functional>
#include <vector>

bool prokhorov_n_rectangular_integration_seq::TestTaskSequential::PreProcessingImpl() {
  if (task_data->inputs.empty() || task_data->inputs_count[0] != 3) {
    return false;
  }

  auto* inputs = reinterpret_cast<double*>(task_data->inputs[0]);

  lower_bound_ = inputs[0];
  upper_bound_ = inputs[1];
  n_ = static_cast<int>(inputs[2]);

  if (lower_bound_ >= upper_bound_) {
    return false;
  }

  return n_ > 0;
}

bool prokhorov_n_rectangular_integration_seq::TestTaskSequential::ValidationImpl() {
  if (task_data->inputs_count[0] != 3) {
    return false;
  }

  if (task_data->outputs_count[0] != 1) {
    return false;
  }

  auto* inputs = reinterpret_cast<double*>(task_data->inputs[0]);
  double lower_bound = inputs[0];
  double upper_bound = inputs[1];
  if (lower_bound >= upper_bound) {
    return false;
  }

  int n = static_cast<int>(inputs[2]);
  return n > 0;
}

bool prokhorov_n_rectangular_integration_seq::TestTaskSequential::RunImpl() {
  result_ = Integrate(f_, lower_bound_, upper_bound_, n_);
  return true;
}

bool prokhorov_n_rectangular_integration_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

void prokhorov_n_rectangular_integration_seq::TestTaskSequential::SetFunction(
    const std::function<double(double)>& func) {
  f_ = func;
}

double prokhorov_n_rectangular_integration_seq::TestTaskSequential::Integrate(const std::function<double(double)>& f,
                                                                              double lower_bound, double upper_bound,
                                                                              int n) {
  double step = (upper_bound - lower_bound) / n;
  double area = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = lower_bound + ((i + 0.5) * step);
    area += f(x) * step;
  }

  return area;
}