#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq {
void bucketRadixSort(std::vector<double>& values);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;
  std::vector<double> output_;

  void sortWithSignHandling(std::vector<double>& values);
  void radixSort(std::vector<double>& values);
};

}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq