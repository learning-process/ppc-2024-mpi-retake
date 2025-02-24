#pragma once

#include <memory>
#include <numeric>
#include <string>

#include "core/task/include/task.hpp"

namespace leontev_n_average_seq {
template <class InOutType>
class VecAvgSequential : public ppc::core::Task {
 public:
  explicit VecAvgSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<InOutType> input_;
  InOutType res{};
};

}  // namespace leontev_n_average_seq
