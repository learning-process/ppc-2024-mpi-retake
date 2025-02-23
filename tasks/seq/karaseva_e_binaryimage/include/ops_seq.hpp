#pragma once

#include <map>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_binaryimage_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int rc_size_{};
  std::map<int, int> label_equivalence_;
  std::vector<int> image_;
  std::vector<int> labeled_image;
  int rows;
  int columns;
};

}  // namespace karaseva_e_binaryimage_seq