#pragma once
#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace sharamygina_i_horizontal_line_filtration_seq {
class horizontal_line_filtration_seq : public ppc::core::Task {
 public:
  explicit horizontal_line_filtration_seq(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  unsigned int gauss[3][3]{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  std::vector<unsigned int> original_data_;
  std::vector<unsigned int> result_data_;
  int rows_;
  int cols_;
  unsigned int InputAnotherPixel(const std::vector<unsigned int>& image, int x, int y, int rows, int cols);
};
}  // namespace sharamygina_i_horizontal_line_filtration_seq