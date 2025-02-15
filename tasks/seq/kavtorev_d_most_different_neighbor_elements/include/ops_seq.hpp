#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kavtorev_d_most_different_neighbor_elements_seq {

class most_different_neighbor_elements_seq : public ppc::core::Task {
 public:
  explicit most_different_neighbor_elements_seq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<int> generator(int sz);

 private:
  std::vector<std::pair<int, int>> input_;
  std::pair<int, int> res{};
};

}  // namespace kavtorev_d_most_different_neighbor_elements_seq