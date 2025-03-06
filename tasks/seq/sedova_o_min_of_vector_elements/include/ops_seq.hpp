// Copyright 2023 Nesterov Alexander
#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sedova_o_min_of_vector_elements_seq {
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::vector<int>> input_;
  int res_{};
};

}  // namespace sedova_o_min_of_vector_elements_seq