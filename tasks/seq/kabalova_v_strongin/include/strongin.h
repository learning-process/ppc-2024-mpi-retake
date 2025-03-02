#pragma once

#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kabalova_v_strongin_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<double(double)> f_)
      : Task(std::move(taskData_)), f(f_) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double left{};
  double right{};
  std::function<double(double)> f;
  std::pair<double, double> result{};
};

}  // namespace kabalova_v_strongin_seq