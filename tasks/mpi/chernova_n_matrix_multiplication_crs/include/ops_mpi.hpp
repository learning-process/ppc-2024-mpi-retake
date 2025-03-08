#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernova_n_matrix_multiplication_crs_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
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
  size_t size_val_A, size_val_B, size_col_A, size_col_B, size_row_A, size_row_B;

  boost::mpi::communicator world;

  void BroadcastMatrixMetadata();
  void InitializeNonRootMatrices();
  void BroadcastMatrixData();
};

}  // namespace chernova_n_matrix_multiplication_crs_mpi
