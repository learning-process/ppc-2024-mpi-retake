// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalinin_d_vector_dot_product_seq {
int vectorDotProduct(const std::vector<int>& v1, const std::vector<int>& v2);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int res{};
  std::vector<std::vector<int>> input_;
};

}  // namespace kalinin_d_vector_dot_product_seq