#pragma once
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace sharamygina_i_vector_dot_product_seq {
class vector_dot_product_seq : public ppc::core::Task {
 public:
  explicit vector_dot_product_seq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> v1;
  std::vector<int> v2;
  int res{};
};
}  // namespace sharamygina_i_vector_dot_product_seq