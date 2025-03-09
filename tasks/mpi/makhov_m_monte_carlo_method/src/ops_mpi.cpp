// Copyright 2023 Nesterov Alexander
#include "mpi/makhov_m_monte_carlo_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "simple_parser.hpp"

bool makhov_m_monte_carlo_method_mpi::TestMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    funcStr_ = *reinterpret_cast<std::string*>(task_data->inputs[0]);
    numSamples_ = *reinterpret_cast<int*>(task_data->inputs[1]);

    auto* limits_ptr = reinterpret_cast<std::pair<double, double>*>(task_data->inputs[2]);
    std::uint32_t dimension = task_data->inputs_count[2];
    limits_.assign(limits_ptr, limits_ptr + dimension);
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
  std::pair<int, int> dummy_pair = {1, 1};
  std::vector<int> dummy_vector = {12};
  boost::mpi::broadcast(world_, numSamples_, 0);
  boost::mpi::broadcast(world_, limits_, 0);
  boost::mpi::broadcast(world_, funcStr_, 0);

  std::regex var_regex("[a-z]");  // Regular expression for variables (a-z)
  std::smatch matches;
  std::string::const_iterator search_start(funcStr_.cbegin());
  std::vector<std::string> variables;

  while (std::regex_search(search_start, funcStr_.cend(), matches, var_regex)) {
    variables.push_back(matches.str(0));
    search_start = matches.suffix().first;
  }

  // Removing duplicates
  std::ranges::sort(variables);
  auto [new_end, end] = std::ranges::unique(variables);
  variables.erase(new_end, variables.end());

  // A container for storing variables and their values
  std::map<std::string, double> var_values;

  // Adding variables to the symbol table
  for (const auto& var : variables) {
    var_values[var] = 0.0;  // Initializing variables to zeros
  }

  std::random_device rd;
  std::mt19937 gen(rd() + world_.rank());  // Unique seed for each process
  std::uniform_real_distribution<> dis(limits_[0].first, limits_[0].second);

  // Calculating the number of points for each process
  int local_samples = numSamples_ / world_.size();
  if (world_.rank() == world_.size() - 1) {
    local_samples += numSamples_ % world_.size();
  }

  double local_sum = 0.0;
  for (int i = 0; i < local_samples; ++i) {  // Generate random point
    // Generating random values ​​for variables
    for (auto& var : var_values) {
      var.second = dis(gen);  // Random value of variable
    }

    // Calculating a function
    SimpleParser parser(funcStr_, var_values);
    local_sum += parser.Parse();
  }

  if (world_.rank() == 0) {
    globalSum_ = local_sum;

    // Accept data from all other processes
    for (int i = 1; i < world_.size(); ++i) {
      double received_sum = NAN;
      world_.recv(i, 0, received_sum);  // Receive data from process i
      globalSum_ += received_sum;       // Summation
    }
  } else {
    // The other processes send their local sums to the root process.
    world_.send(0, 0, local_sum);  // Sending data to process 0
  }

  if (world_.rank() == 0) {
    double volume = 1.0;
    for (const auto& limit : limits_) {
      volume *= (limit.second - limit.first);
    }
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
