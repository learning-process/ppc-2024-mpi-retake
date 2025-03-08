#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasenkov_a_gauss_jordan_seq {

class GaussJordanMethodSequential : public ppc::core::Task {
 private:
  int n_size_;
  std::vector<double> sys_matrix_;

 public:
  explicit GaussJordanMethodSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasenkov_a_gauss_jordan_seq