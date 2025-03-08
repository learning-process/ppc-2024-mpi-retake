// Copyright 2023 Nesterov Alexander
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalinin_d_vector_dot_product_seq {
int VectorDotProduct(const std::vector<int>& v1, const std::vector<int>& v2);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int res_{};
  std::vector<std::vector<int>> input_;
};

}  // namespace kalinin_d_vector_dot_product_seq