#pragma once

#include <utility>
#include <vector>
#include <cstdint>
#include <functional>

#include "core/task/include/task.hpp"

namespace makhov_m_monte_carlo_method_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::function<double(const std::vector<double>&)> func;
  std::vector<std::pair<double, double>> limits;
  int numSamples{};
  double answer{};
  uint8_t* answerDataPtr{};
};

}  // namespace makhov_m_monte_carlo_method_seq