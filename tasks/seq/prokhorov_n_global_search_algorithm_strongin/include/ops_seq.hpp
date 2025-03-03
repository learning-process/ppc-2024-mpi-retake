#pragma once

#include <functional>
#include <utility>

#include "core/task/include/task.hpp"

namespace prokhorov_n_global_search_algorithm_strongin_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data, std::function<double(double)> func)
      : Task(std::move(task_data)), f_(std::move(func)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double a_{};
  double b_{};
  double epsilon_{};
  double result_{};

  std::function<double(double)> f_;
  double StronginAlgorithm();
};

}  // namespace prokhorov_n_global_search_algorithm_strongin_seq