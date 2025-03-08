#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_num_of_alternations_signs_seq {

class AlternatingSignsSequential : public ppc::core::Task {
 public:
  explicit AlternatingSignsSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int alternations_count_{0};
};

}  // namespace karaseva_e_num_of_alternations_signs_seq