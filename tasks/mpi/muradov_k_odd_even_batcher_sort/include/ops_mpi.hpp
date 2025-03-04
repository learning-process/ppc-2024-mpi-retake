#ifndef MURADOV_K_ODD_EVEN_BATCHER_SORT_OPS_MPI_HPP
#define MURADOV_K_ODD_EVEN_BATCHER_SORT_OPS_MPI_HPP

#include <vector>

#include "core/task/include/task.hpp"

namespace muradov_k_odd_even_batcher_sort {

void QSort(std::vector<int>& v, int l, int r);
void OddEvenBatcherSort(std::vector<int>& v);

class OddEvenBatcherSortTask : public ppc::core::Task {
 public:
  explicit OddEvenBatcherSortTask(ppc::core::TaskDataPtr task_data) : ppc::core::Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> data_;
};

}  // namespace muradov_k_odd_even_batcher_sort

#endif  // MURADOV_K_ODD_EVEN_BATCHER_SORT_OPS_MPI_HPP