#ifndef MURADOV_K_RADIX_SORT_OPS_SEQ_HPP
#define MURADOV_K_RADIX_SORT_OPS_SEQ_HPP

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace muradov_k_radix_sort {

void RadixSort(std::vector<int>& v);

class RadixSortTask : public ppc::core::Task {
 public:
  explicit RadixSortTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> data_;
};

}  // namespace muradov_k_radix_sort

#endif