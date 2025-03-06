// Copyright 2023 Nesterov Alexander
#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_sum_values_by_cols_seq {
class SumValByCols : public ppc::core::Task {
 public:
  explicit SumValByCols(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  unsigned int rows_, cols_;
  std::vector<int> sum_;
};

}  // namespace dudchenko_o_sum_values_by_cols_seq