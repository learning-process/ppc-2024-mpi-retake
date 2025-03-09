// Copyright 2023 Nesterov Alexander
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chastov_v_algorithm_cannon_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  size_t matrix_size_{}, total_elements_{};
  std::vector<double> first_matrix_, second_matrix_, result_matrix_;
};

}  // namespace chastov_v_algorithm_cannon_seq