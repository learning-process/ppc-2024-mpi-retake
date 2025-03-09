#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sedova_o_test_task_seq {
template <typename T>
T MultVectors(const std::vector<T> &vector_A, const std::vector<T> &vector_B);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> Convertirovanie(
    const std::vector<double> &A, const std::vector<int> &row_ind_A, const std::vector<int> &col_ind_A,
    const std::vector<double> &B, const std::vector<int> &row_ind_B, const std::vector<int> &col_ind_B, int rows_A,
    int cols_A, int rows_B, int cols_B);
void ConvertToCCS(const std::vector<std::vector<double>> &matrix, std::vector<double> &values,
                  std::vector<int> &row_indices, std::vector<int> &col_pointers);
void FillData(std::shared_ptr<ppc::core::TaskData> &taskData, int rows_A, int cols_A, int rows_B, int cols_B,
              std::vector<double> &A, std::vector<int> &row_ind_A, std::vector<int> &col_ind_A, std::vector<double> &B,
              std::vector<int> &row_ind_B, std::vector<int> &col_ind_B, std::vector<std::vector<double>> &out);
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::vector<double>> ans;
  std::vector<double> A, B;
  std::vector<int> row_ind_A, row_ind_B, col_ind_A, col_ind_B;
  int rows_A, rows_B, cols_A, cols_B, size_A, size_B, row_ind_size_A, row_ind_size_B, col_ind_size_A, col_ind_size_B;
};
}  // namespace nesterov_a_test_task_seq