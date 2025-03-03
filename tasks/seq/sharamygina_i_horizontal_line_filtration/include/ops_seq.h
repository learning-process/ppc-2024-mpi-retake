#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sharamygina_i_horizontal_line_filtration_seq {
class HorizontalLineFiltrationSeq : public ppc::core::Task {
 public:
  explicit HorizontalLineFiltrationSeq(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  unsigned int gauss_[3][3]{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  std::vector<unsigned int> original_data_;
  std::vector<unsigned int> result_data_;
  int rows_;
  int cols_;
  unsigned int InputAnotherPixel(const std::vector<unsigned int>& image, int x, int y, int rows, int cols);
};
}  // namespace sharamygina_i_horizontal_line_filtration_seq