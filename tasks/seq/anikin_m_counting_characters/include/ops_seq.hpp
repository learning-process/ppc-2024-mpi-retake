// Anikin Maksim 2025
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_counting_characters_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<char> input_1_, input_2_;
  int res_;
};

}  // namespace anikin_m_counting_characters_seq