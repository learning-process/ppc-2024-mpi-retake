#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace ersoz_b_test_task_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::vector<char>> input_image_;
  std::vector<std::vector<char>> output_image_;
  int img_size_{};
  double sigma_{0.5};
};

}  // namespace ersoz_b_test_task_seq
