#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace budazhapova_e_count_freq_character_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::string input_;
  char symb{};
  int res{};
};

}  // namespace budazhapova_e_count_freq_character_seq