#include "mpi/khokhlov_a_multi_integration_monte_karlo/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <random>
#include <vector>

bool khokhlov_a_multi_integration_monte_karlo_mpi::MonteCarloMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    dimension_ = task_data->inputs_count[0];
    lower_bound_ = std::vector<double>(dimension_);
    auto* lbound = reinterpret_cast<double*>(task_data->inputs[0]);
    std::copy(lbound, lbound + dimension_, lower_bound_.data());
    upper_bound_ = std::vector<double>(dimension_);
    auto* ubound = reinterpret_cast<double*>(task_data->inputs[1]);
    std::copy(ubound, ubound + dimension_, upper_bound_.data());
    N_ = task_data->inputs_count[1];
    result_ = 0.0;
  }
  return true;
}

bool khokhlov_a_multi_integration_monte_karlo_mpi::MonteCarloMpi::ValidationImpl() {
  if (world_.rank() == 0) {
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
  }
  return true;
}

bool khokhlov_a_multi_integration_monte_karlo_mpi::MonteCarloMpi::RunImpl() {
  double volume = 0.0;
  broadcast(world_, dimension_, 0);
  broadcast(world_, N_, 0);
  lower_bound_.resize(dimension_);
  upper_bound_.resize(dimension_);
  broadcast(world_, lower_bound_.data(), (int)dimension_, 0);
  broadcast(world_, upper_bound_.data(), (int)dimension_, 0);

  if (world_.rank() == 0) {
    volume = 1.0 / N_;
    for (unsigned int i = 0; i < dimension_; i++) {
      volume *= (upper_bound_[i] - lower_bound_[i]);
    }
  }
  broadcast(world_, volume, 0);

  int delta = (int)N_ / world_.size();
  int last = (int)N_ % world_.size();
  std::random_device rd;
  std::mt19937 gen(world_.rank() + rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  double local_res = 0.0;
  if (last != 0) {
    delta++;
  }
  for (int i = 0; i < delta; i++) {
    std::vector<double> x(dimension_, 1);
    for (unsigned int j = 0; j < dimension_; j++) {
      x[j] = lower_bound_[j] + (upper_bound_[j] - lower_bound_[j]) * dis(gen);
    }
    local_res += integrand(x);
  }
  local_res *= volume;
  boost::mpi::reduce(world_, local_res, result_, std::plus<>(), 0);
  return true;
}

bool khokhlov_a_multi_integration_monte_karlo_mpi::MonteCarloMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  }
  return true;
}