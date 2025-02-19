#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace budazhapova_betcher_odd_even_merge_seq {

class MergeSequential : public ppc::core::Task {
 public:
  explicit MergeSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessing() override;
  bool Validation() override;
  bool Run() override;
  bool PostProcessing() override;

 private:
  std::vector<int> res_;
  int n_el_ = 0;
};
}  // namespace budazhapova_betcher_odd_even_merge_seq