#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_linear_hist_stretch_seq {

class LinearHistStretchSeq : public ppc::core::Task {
 public:
  explicit LinearHistStretchSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> input_;
  std::vector<uint8_t> output_;
  uint8_t min_val_, max_val_;
};

}  // namespace konkov_i_linear_hist_stretch_seq
