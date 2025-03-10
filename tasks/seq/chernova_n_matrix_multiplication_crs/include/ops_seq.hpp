#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernova_n_matrix_multiplication_crs_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  struct SparseMatrixCRS {
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
  };

  SparseMatrixCRS matrixA, matrixB, resultMatrix;
  int rowsA, colsA, rowsB, colsB, rowsRes, colsRes;
};

}  // namespace chernova_n_matrix_multiplication_crs_seq