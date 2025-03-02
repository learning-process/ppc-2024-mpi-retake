#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> values;
  int size = 0;

  static void SortValues(std::vector<double>& values);
  static void RadixSort(std::vector<uint64_t>& keys);
};

}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq