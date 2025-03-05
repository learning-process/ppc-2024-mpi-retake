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
  int rowsA, colsA, rowsB, colsB, rowsAt, colsAt;
  std::vector<std::vector<double>> A, B;
  std::vector<double> AVal, BVal, AtVal, resVal;
  std::vector<int> ARowInd, AColPtr, BRowInd, BColPtr, AtRowInd, AtColPtr, resInd, resPtr;
};

}  // namespace sedova_o_multiply_matrices_ccs_seq