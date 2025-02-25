// Copyright 2023 Nesterov Alexander
#pragma once

#include <limits>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace sedova_o_min_of_vector_elements_seq {

std::vector<int> getRandomVector(int sz, int min = 0, int max = 100);
std::vector<std::vector<int>> getRandomMatrix(int rows, int columns, int min = 0, int max = 100);

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