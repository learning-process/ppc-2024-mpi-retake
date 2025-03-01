#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace makadrai_a_sobel_seq {

class Sobel : public ppc::core::Task {
 public:
  explicit Sobel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int height_img_;
  int width_img_;
  int peding_ = 2;

  std::vector<int> img_;
  std::vector<int> simg_;
};

}  // namespace makadrai_a_sobel_seq