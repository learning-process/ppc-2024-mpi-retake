#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

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
  std::function<double(const std::vector<double>&)> func_;
  std::vector<std::pair<double, double>> limits_;
  int numSamples_{};
  double answer_{};
  uint8_t* answerDataPtr_{};
};

}  // namespace makhov_m_monte_carlo_method_seq