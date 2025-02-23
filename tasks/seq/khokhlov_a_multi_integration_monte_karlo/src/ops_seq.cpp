#include "seq/khokhlov_a_multi_integration_monte_karlo/include/ops_seq.hpp"

#include <algorithm>
#include <random>
#include <vector>

bool khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq::PreProcessingImpl() {
  dimension_ = task_data->inputs_count[0];
  lower_bound_ = std::vector<double>(dimension_);
  auto* lbound = reinterpret_cast<double*>(task_data->inputs[0]);
  std::copy(lbound, lbound + dimension_, lower_bound_.data());
  upper_bound_ = std::vector<double>(dimension_);
  auto* ubound = reinterpret_cast<double*>(task_data->inputs[1]);
  std::copy(ubound, ubound + dimension_, upper_bound_.data());
  N_ = task_data->inputs_count[1];
  return true;
}

bool khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq::ValidationImpl() {
  if (task_data->inputs_count[0] < 1 || task_data->inputs_count[1] < 1) {
    return false;
  }
  if (task_data->inputs_count[2] != task_data->inputs_count[3]) {
    return false;
  }
  auto* lbound = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* ubound = reinterpret_cast<double*>(task_data->inputs[1]);
  if (lbound == nullptr || ubound == nullptr) {
    return false;
  }
  for (unsigned int i = 0; i < task_data->inputs_count[0]; i++) {
    if (lbound[i] > ubound[i]) {
      return false;
    }
  }
  return true;
}

bool khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq::RunImpl() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  result_ = 0.0;
  for (unsigned int i = 0; i < N_; i++) {
    std::vector<double> x(dimension_);
    for (unsigned int j = 0; j < dimension_; j++) {
      x[j] = lower_bound_[j] + (upper_bound_[j] - lower_bound_[j]) * dis(gen);
    }
    result_ += integrand(x);
  }
  double volume = 1.0 / N_;
  for (unsigned int i = 0; i < dimension_; i++) {
    volume *= (upper_bound_[i] - lower_bound_[i]);
  }
  result_ *= volume;
  return true;
}

bool khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}