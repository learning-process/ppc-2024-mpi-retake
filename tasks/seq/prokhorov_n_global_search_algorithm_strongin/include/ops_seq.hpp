#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <utility>

#include "core/task/include/task.hpp"

namespace prokhorov_n_global_search_algorithm_strongin_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> task_data, std::function<double(double)> f)
      : Task(std::move(task_data)), f_(std::move(f)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double stronginAlgorithm(double a, double b, double eps, double r, const std::function<double(double)>& f);

 private:
  double a_{};
  double b_{};
  std::function<double(double)> f_;
  double result_;
};

}  // namespace prokhorov_n_global_search_algorithm_strongin_seq