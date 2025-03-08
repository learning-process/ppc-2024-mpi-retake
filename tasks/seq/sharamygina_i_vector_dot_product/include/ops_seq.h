#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sharamygina_i_vector_dot_product_seq {
class VectorDotProductSeq : public ppc::core::Task {
 public:
  explicit VectorDotProductSeq(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> v1_;
  std::vector<int> v2_;
  int res_{};
};
}  // namespace sharamygina_i_vector_dot_product_seq