// Copyright 2023 Nesterov Alexander
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace shishkarev_a_dijkstra_algorithm_seq {

void convertToCRS(const std::vector<int>& w, std::vector<int>& values, std::vector<int>& colIndex,
                  std::vector<int>& rowPtr, int n);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_;
  int st{};
  int size{};
};

}  // namespace shishkarev_a_dijkstra_algorithm_seq