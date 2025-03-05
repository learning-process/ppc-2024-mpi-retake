#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_multi_integration_monte_karlo_seq {

class MonteCarloSeq : public ppc::core::Task {
 public:
  explicit MonteCarloSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::function<double(const std::vector<double>&)> integrand;

 private:
  unsigned int dimension_;
  unsigned int N_;
  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;
  double result_;
};

}  // namespace khokhlov_a_multi_integration_monte_karlo_seq