#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace leontev_n_binary_seq {

class BinarySegmentsSeq : public ppc::core::Task {
 public:
  explicit BinarySegmentsSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  [[nodiscard]] size_t GetIndex(size_t i, size_t j) const;
  void LoopProcess(size_t col, size_t row, uint32_t& cur_label, std::vector<std::set<uint32_t>>& label_equivalences);
  std::vector<uint8_t> input_image_;
  std::vector<uint32_t> labels_;
  size_t rows_;
  size_t cols_;
};
}  // namespace leontev_n_binary_seq
