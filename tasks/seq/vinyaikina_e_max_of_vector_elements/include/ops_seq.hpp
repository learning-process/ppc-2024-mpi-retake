
#pragma once

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vinyaikina_e_max_of_vector_elements_seq {

class VectorMaxSeq : public ppc::core::Task {
 public:
  explicit VectorMaxSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int32_t> input_;
  int32_t max_ = std::numeric_limits<int32_t>::min();
};

}  // namespace vinyaikina_e_max_of_vector_elements_seq
