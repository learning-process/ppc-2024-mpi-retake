// Copyright 2023 Nesterov Alexander
#include "mpi/makhov_m_monte_carlo_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "simple_parser.hpp"

bool makhov_m_monte_carlo_method_mpi::TestMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    funcStr_ = *reinterpret_cast<std::string*>(task_data->inputs[0]);
    numSamples_ = *reinterpret_cast<int*>(task_data->inputs[1]);

    auto* limits_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
    limits_[0] = limits_ptr[0];
    limits_[1] = limits_ptr[1];

    std::regex var_regex("[a-z]");
    std::smatch matches;
    std::string::const_iterator search_start(funcStr_.cbegin());
    std::vector<std::string> variables;

    while (std::regex_search(search_start, funcStr_.cend(), matches, var_regex)) {
      variables.push_back(matches.str(0));
      search_start = matches.suffix().first;
    }

    std::sort(variables.begin(), variables.end());
    variables.erase(std::unique(variables.begin(), variables.end()), variables.end());
    dimension_ = variables.size();
  }
  return true;
}

bool makhov_m_monte_carlo_method_mpi::TestMPITaskParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count.size() == 3 && task_data->outputs_count.size() == 1 &&
           task_data->outputs_count[0] == task_data->inputs_count[0];
  }
  return true;
}

bool makhov_m_monte_carlo_method_mpi::TestMPITaskParallel::RunImpl() {
  globalSum_ = 0.0;
  if (world_.size() > 1) {
    boost::mpi::broadcast(world_, numSamples_, 0);
    boost::mpi::broadcast(world_, limits_, 0);
    boost::mpi::broadcast(world_, funcStr_, 0);
    boost::mpi::broadcast(world_, dimension_, 0);
  }

  std::regex var_regex("[a-z]");
  std::smatch matches;
  std::string::const_iterator search_start(funcStr_.cbegin());
  std::vector<std::string> variables;

  while (std::regex_search(search_start, funcStr_.cend(), matches, var_regex)) {
    variables.push_back(matches.str(0));
    search_start = matches.suffix().first;
  }

  std::sort(variables.begin(), variables.end());
  variables.erase(std::unique(variables.begin(), variables.end()), variables.end());

  std::map<std::string, double> var_values;
  for (const auto& var : variables) {
    var_values[var] = 0.0;
  }

  std::random_device rd;
  std::mt19937 gen(rd() + world_.rank());
  std::uniform_real_distribution<> dis(limits_[0], limits_[1]);

  int local_samples = numSamples_ / world_.size();
  if (world_.rank() == world_.size() - 1) {
    local_samples += numSamples_ % world_.size();
  }

  double local_sum = 0.0;
  for (int i = 0; i < local_samples; ++i) {
    for (auto& var : var_values) {
      var.second = dis(gen);
    }

    SimpleParser parser(funcStr_, var_values);
    local_sum += parser.Parse();
  }

  boost::mpi::reduce(world_, local_sum, globalSum_, std::plus<double>(), 0);

  if (world_.rank() == 0) {
    double volume = limits_[1] - limits_[0];
    answer_ = volume * (globalSum_ / numSamples_);
    answerDataPtr_ = new uint8_t[sizeof(double)];
    std::memcpy(answerDataPtr_, &answer_, sizeof(double));
  }

  world_.barrier();
  return true;
}

bool makhov_m_monte_carlo_method_mpi::TestMPITaskParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    task_data->outputs_count = task_data->inputs_count;
    task_data->outputs[0] = answerDataPtr_;
    delete[] answerDataPtr_;
  }
  return true;
}
