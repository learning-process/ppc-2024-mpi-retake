// Copyright 2023 Nesterov Alexander
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shishkarev_a_sum_of_vector_elements_seq {
template <class InOutType>
class VectorSumSequential : public ppc::core::Task {
 public:
  explicit VectorSumSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<InOutType> input_data_;
  InOutType result_;
};

}  // namespace shishkarev_a_sum_of_vector_elements_seq
