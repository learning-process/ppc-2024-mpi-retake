#include "seq/prokhorov_n_global_search_algorithm_strongin/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace prokhorov_n_global_search_algorithm_strongin_seq {

bool TestTaskSequential::PreProcessingImpl() {
  if (task_data->inputs_count[0] == 0 || task_data->inputs_count[1] == 0 || task_data->inputs_count[2] == 0) {
    return false;
  }

  a_ = *reinterpret_cast<double*>(task_data->inputs[0]);
  b_ = *reinterpret_cast<double*>(task_data->inputs[1]);
  epsilon_ = *reinterpret_cast<double*>(task_data->inputs[2]);

  return true;
}

bool TestTaskSequential::ValidationImpl() {
  if (task_data->inputs_count[0] == 0 || task_data->inputs_count[1] == 0 || task_data->inputs_count[2] == 0) {
    return false;
  }

  return (a_ < b_) && (epsilon_ > 0);
}

bool TestTaskSequential::RunImpl() {
  result_ = StronginAlgorithm();
  return true;
}

bool TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

double TestTaskSequential::StronginAlgorithm() {
  std::vector<double> y = {a_, b_};
  std::vector<double> z = {f_(a_), f_(b_)};
  int n = y.size() - 1;
  double M, m, r = 2.0;
  const int max_iterations = 1000;
  int iteration = 0;

  while (iteration < max_iterations) {
    iteration++;

    M = 0.0;
    for (int i = 1; i <= n; ++i) {
      double delta = std::abs((z[i] - z[i - 1]) / (y[i] - y[i - 1]));
      if (delta > M) M = delta;
    }
    m = (M == 0) ? 1 : r * M;

    std::vector<double> R(n);
    for (int i = 1; i <= n; ++i) {
      R[i - 1] = m * (y[i] - y[i - 1]) + std::pow(z[i] - z[i - 1], 2) / (m * (y[i] - y[i - 1])) - 2 * (z[i] + z[i - 1]);
    }

    auto max_R = std::max_element(R.begin(), R.end());
    int s = std::distance(R.begin(), max_R);
    double tau = (y[s + 1] + y[s]) / 2 - (z[s + 1] - z[s]) / (2 * m);
    y.insert(y.begin() + s + 1, tau);
    z.insert(z.begin() + s + 1, f_(tau));
    n++;

    double max_interval = 0.0;
    for (int i = 1; i <= n; ++i) {
      double interval = y[i] - y[i - 1];
      if (interval > max_interval) max_interval = interval;
    }
    if (max_interval < epsilon_) {
      auto min_z = std::min_element(z.begin(), z.end());
      return y[std::distance(z.begin(), min_z)];
    }
  }

  auto min_z = std::min_element(z.begin(), z.end());
  return y[std::distance(z.begin(), min_z)];
}
}  // namespace prokhorov_n_global_search_algorithm_strongin_seq