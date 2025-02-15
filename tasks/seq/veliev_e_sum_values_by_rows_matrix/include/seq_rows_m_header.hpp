#pragma once
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace veliev_e_sum_values_by_rows_matrix_seq {
class SumValuesByRowsMatrixSeq : public ppc::core::Task {
 public:
  explicit SumValuesByRowsMatrixSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int elem_total_, cols_total_, rows_total_;
};
void GetRndMatrix(std::vector<int>& vec);
void SeqProcForChecking(std::vector<int>& vec, int rows_size, std::vector<int>& output);
}  // namespace veliev_e_sum_values_by_rows_matrix_seq