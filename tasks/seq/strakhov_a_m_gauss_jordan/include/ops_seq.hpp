#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace strakhov_a_m_gauss_jordan_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
  size_t row_size_, col_size_;
};

}  // namespace strakhov_a_m_gauss_jordan_seq