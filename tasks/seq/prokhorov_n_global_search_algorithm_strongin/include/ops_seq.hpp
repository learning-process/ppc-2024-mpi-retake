#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace prokhorov_n_global_search_algorithm_strongin_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double a{};
  double b{};
  double epsilon{};
  double result{};

  std::function<double(double)> f;
  double stronginAlgorithm();
};

}  // namespace prokhorov_n_global_search_algorithm_strongin_seq