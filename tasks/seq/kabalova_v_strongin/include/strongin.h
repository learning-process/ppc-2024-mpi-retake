#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <utility>

#include "core/task/include/task.hpp"

namespace kabalova_v_strongin_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> task_data, std::function<double(double)> f)
      : Task(std::move(task_data)), f_(std::move(f)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double left_{};
  double right_{};
  std::function<double(double)> f_;
  std::pair<double, double> result_;
};

}  // namespace kabalova_v_strongin_seq