#include "seq/makhov_m_monte_carlo_method/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <utility>
#include <vector>

bool makhov_m_monte_carlo_method_seq::TestTaskSequential::PreProcessingImpl() {
  func_ = *reinterpret_cast<std::function<double(const std::vector<double>&)>*>(task_data->inputs[0]);
  numSamples_ = *reinterpret_cast<int*>(task_data->inputs[1]);

  auto* limits_ptr = reinterpret_cast<std::pair<double, double>*>(task_data->inputs[2]);
  std::uint32_t dimension = task_data->inputs_count[2];
  limits_.assign(limits_ptr, limits_ptr + dimension);
  return true;
}

bool makhov_m_monte_carlo_method_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count.size() == 3 && task_data->outputs_count.size() == 1 &&
         task_data->outputs_count[0] == task_data->inputs_count[0];
}

bool makhov_m_monte_carlo_method_seq::TestTaskSequential::RunImpl() {
  std::random_device rd;
  std::mt19937 gen(rd());

  // Vector of distributions for each variable
  std::vector<std::uniform_real_distribution<>> distributions;
  for (const auto& limit : limits_) {
    distributions.emplace_back(limit.first, limit.second);
  }
  double sum = 0.0;

  // Generating random points and calculating the sum of function values
  for (int i = 0; i < numSamples_; ++i) {
    std::vector<double> point;
    for (size_t j = 0; j < limits_.size(); ++j) {
      point.push_back(distributions[j](gen));  // Generate random point
    }
    sum += func_(point);  // Add the function value to the sum
  }

  // Calculating the volume of the integration region
  double volume = 1.0;
  for (const auto& limit : limits_) {
    volume *= (limit.second - limit.first);  // Volume = product of interval lengths
  }

  // Estimation of the integral
  answer_ = volume * (sum / numSamples_);  // Integral = volume * mean
  answerDataPtr_ = new uint8_t[sizeof(double)];
  std::memcpy(answerDataPtr_, &answer_, sizeof(double));
  return true;
}

bool makhov_m_monte_carlo_method_seq::TestTaskSequential::PostProcessingImpl() {
  task_data->outputs_count = task_data->inputs_count;
  task_data->outputs[0] = answerDataPtr_;
  delete[] answerDataPtr_;
  return true;
}
