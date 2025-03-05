#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sedova_o_multiply_matrices_ccs_seq {
inline void Convertirovanie(const std::vector<std::vector<double>>& matrix, int rows, int cols,
                            std::vector<double>& values, std::vector<int>& rowIndices, std::vector<int>& colPtr) {
  colPtr.clear();
  colPtr.push_back(0);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      if (matrix[i][j] != 0.0) {
        values.push_back(matrix[i][j]);
        rowIndices.push_back(i);
      }
    }
    colPtr.push_back(values.size());
  }
}

inline void Transponirovanie(const std::vector<double>& values, const std::vector<int>& rowIndices,
                             const std::vector<int>& colPtr, int rows, int cols, std::vector<double>& tValues,
                             std::vector<int>& tRowIndices, std::vector<int>& tColPtr) {
  std::vector<std::vector<int>> intVectors(rows);
  std::vector<std::vector<double>> realVectors(rows);

  for (int col = 0; col < cols; ++col) {
    for (int i = colPtr[col]; i < colPtr[col + 1]; ++i) {
      int row = rowIndices[i];
      double value = values[i];

      intVectors[row].push_back(col);
      realVectors[row].push_back(value);
    }
  }

  tColPtr.clear();
  tValues.clear();
  tRowIndices.clear();

  tColPtr.push_back(0);
  for (int i = 0; i < rows; ++i) {
    for (size_t j = 0; j < intVectors[i].size(); ++j) {
      tRowIndices.push_back(intVectors[i][j]);
      tValues.push_back(realVectors[i][j]);
    }
    tColPtr.push_back(tValues.size());
  }
}

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int rows_A_, cols_A_, rows_B_, cols_B_, rows_At_, cols_At_;
  std::vector<std::vector<double>> A_, B_;
  std::vector<double> A_val_, B_val_, At_val_, res_val_;
  std::vector<int> A_row_ind_, A_col_ptr_, B_row_ind_, B_col_ptr_, At_row_ind_, At_col_ptr_, res_ind_, res_ptr_;
};

}  // namespace sedova_o_multiply_matrices_ccs_seq