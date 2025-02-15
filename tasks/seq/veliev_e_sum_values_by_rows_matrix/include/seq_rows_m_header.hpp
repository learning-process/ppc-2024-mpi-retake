#pragma once
#include <numeric>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace veliev_e_sum_values_by_rows_matrix_seq {
class sum_values_by_rows_matrix_seq : public ppc::core::Task {
 public:
  explicit sum_values_by_rows_matrix_seq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int elem_total, cols_total, rows_total;
};
void get_rnd_matrix(std::vector<int>& vec);
void seq_proc_for_checking(std::vector<int>& vec, int rows_size, std::vector<int>& output);
}  // namespace veliev_e_sum_values_by_rows_matrix_seq