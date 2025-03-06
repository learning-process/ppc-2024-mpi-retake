#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace markin_i_rectangle_method_seq {
  float f(float x);

class RectangleSequential : public ppc::core::Task {
 public:
  explicit RectangleSequential(ppc::core::TaskDataPtr task_data)
  : Task(std::move(task_data)),
    output_(0.0f),
    left_(0.0f),
    right_(0.0f),
    steps_(0)
{}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;


 private:
  float output_, left_, right_;
  int steps_;
};

}