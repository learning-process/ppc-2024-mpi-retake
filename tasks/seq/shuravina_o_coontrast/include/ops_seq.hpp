#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_contrast {

class ContrastTaskSequential : public ppc::core::Task {
 public:
  explicit ContrastTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> input_;
  std::vector<uint8_t> output_;
  int width_{}, height_{};

  void IncreaseContrast();
};

}  // namespace shuravina_o_contrast