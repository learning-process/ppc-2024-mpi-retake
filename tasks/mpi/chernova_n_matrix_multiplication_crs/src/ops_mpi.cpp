#include "mpi/chernova_n_matrix_multiplication_crs/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

bool chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world.rank() == 0) {
    if (task_data->inputs.size() < 6 || task_data->inputs_count.size() < 6 || task_data->outputs.size() < 3) {
      throw std::runtime_error("Insufficient data in task_data");
    }

    auto* values_a = reinterpret_cast<double*>(task_data->inputs[0]);
    auto* col_indices_a = reinterpret_cast<int*>(task_data->inputs[1]);
    auto* row_ptr_a = reinterpret_cast<int*>(task_data->inputs[2]);

    auto* values_b = reinterpret_cast<double*>(task_data->inputs[3]);
    auto* col_indices_b = reinterpret_cast<int*>(task_data->inputs[4]);
    auto* row_ptr_b = reinterpret_cast<int*>(task_data->inputs[5]);

    matrixA.values.assign(values_a, values_a + task_data->inputs_count[0]);
    matrixA.col_indices.assign(col_indices_a, col_indices_a + task_data->inputs_count[1]);
    matrixA.row_ptr.assign(row_ptr_a, row_ptr_a + task_data->inputs_count[2]);

    matrixB.values.assign(values_b, values_b + task_data->inputs_count[3]);
    matrixB.col_indices.assign(col_indices_b, col_indices_b + task_data->inputs_count[4]);
    matrixB.row_ptr.assign(row_ptr_b, row_ptr_b + task_data->inputs_count[5]);

    rowsA = static_cast<int>(task_data->inputs_count[2] - 1);
    rowsB = static_cast<int>(task_data->inputs_count[5] - 1);
    colsA = *std::ranges::max_element(matrixA.col_indices) + 1;
    colsB = *std::ranges::max_element(matrixB.col_indices) + 1;

    rowsRes = rowsA;
    colsRes = colsB;
    resultMatrix.row_ptr.assign(rowsRes + 1, 0);

    size_val_A = matrixA.values.size();
    size_val_B = matrixB.values.size();
    size_col_A = matrixA.col_indices.size();
    size_col_B = matrixB.col_indices.size();
    size_row_A = matrixA.row_ptr.size();
    size_row_B = matrixB.row_ptr.size();
  }
  return true;
}

bool chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::ValidationImpl() {
  if (world.rank() == 0) {
    colsA = *std::max_element(reinterpret_cast<int*>(task_data->inputs[1]),
                              reinterpret_cast<int*>(task_data->inputs[1]) + task_data->inputs_count[1]) +
            1;
    rowsB = static_cast<int>(task_data->inputs_count[5] - 1);
    return colsA == rowsB;
  }
  return true;
}

bool chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::RunImpl() {
  BroadcastMatrixMetadata();
  InitializeNonRootMatrices();
  BroadcastMatrixData();

  int number_of_rows = rowsRes / world.size();
  int start_row = world.rank() * number_of_rows;
  int end_row = (world.rank() == world.size() - 1) ? rowsRes : start_row + number_of_rows;

  std::vector<double> temp(colsRes, 0.0);
  std::vector<double> local_values;
  std::vector<int> local_col_index;
  std::vector<int> local_row_ptr(end_row - start_row + 1, 0);
  int local_nnz_count = 0;
  for (int i = start_row; i < end_row; i++) {
    std::ranges::fill(temp, 0.0);
    for (int index_a = matrixA.row_ptr[i]; index_a < matrixA.row_ptr[i + 1]; index_a++) {
      double a_val = matrixA.values[index_a];
      int a_col = matrixA.col_indices[index_a];
      for (int index_b = matrixB.row_ptr[a_col]; index_b < matrixB.row_ptr[a_col + 1]; index_b++) {
        int b_col = matrixB.col_indices[index_b];
        temp[b_col] += a_val * matrixB.values[index_b];
      }
    }

    local_row_ptr[i - start_row] = local_nnz_count;
    for (int column = 0; column < colsRes; ++column) {
      if (temp[column] != 0.0) {
        local_values.push_back(temp[column]);
        local_col_index.push_back(column);
        local_nnz_count++;
      }
    }
  }
  local_row_ptr[end_row - start_row] = local_nnz_count;
  if (world.rank() == 0) {
    std::vector<std::vector<double>> all_values(world.size());
    std::vector<std::vector<int>> all_col_indices(world.size());
    std::vector<std::vector<int>> all_row_ptrs(world.size());

    boost::mpi::gather(world, local_values, all_values, 0);
    boost::mpi::gather(world, local_col_index, all_col_indices, 0);
    boost::mpi::gather(world, local_row_ptr, all_row_ptrs, 0);
    for (int i = 0; i < world.size(); ++i) {
      resultMatrix.values.insert(resultMatrix.values.end(), all_values[i].begin(), all_values[i].end());
      resultMatrix.col_indices.insert(resultMatrix.col_indices.end(), all_col_indices[i].begin(),
                                      all_col_indices[i].end());
      if (i == 0) {
        resultMatrix.row_ptr = all_row_ptrs[i];
      } else {
        int offset = resultMatrix.row_ptr.back();
        for (size_t j = 1; j < all_row_ptrs[i].size(); ++j) {
          resultMatrix.row_ptr.push_back(all_row_ptrs[i][j] + offset);
        }
      }
    }
  } else {
    boost::mpi::gather(world, local_values, 0);
    boost::mpi::gather(world, local_col_index, 0);
    boost::mpi::gather(world, local_row_ptr, 0);
  }
  return true;
}

void chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::BroadcastMatrixMetadata() {
  boost::mpi::broadcast(world, size_val_A, 0);
  boost::mpi::broadcast(world, size_val_B, 0);
  boost::mpi::broadcast(world, size_col_A, 0);
  boost::mpi::broadcast(world, size_col_B, 0);
  boost::mpi::broadcast(world, size_row_A, 0);
  boost::mpi::broadcast(world, size_row_B, 0);
}

void chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::InitializeNonRootMatrices() {
  if (world.rank() != 0) {
    matrixA.values = std::vector<double>(size_val_A, 0.0);
    matrixB.values = std::vector<double>(size_val_B, 0.0);
    matrixA.col_indices = std::vector<int>(size_col_A, 0);
    matrixB.col_indices = std::vector<int>(size_col_B, 0);
    matrixA.row_ptr = std::vector<int>(size_row_A, 0);
    matrixB.row_ptr = std::vector<int>(size_row_B, 0);
  } else {
    resultMatrix.values.clear();
    resultMatrix.col_indices.clear();
  }
}

void chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::BroadcastMatrixData() {
  boost::mpi::broadcast(world, matrixA.values, 0);
  boost::mpi::broadcast(world, matrixA.col_indices, 0);
  boost::mpi::broadcast(world, matrixA.row_ptr, 0);
  boost::mpi::broadcast(world, matrixB.values, 0);
  boost::mpi::broadcast(world, matrixB.col_indices, 0);
  boost::mpi::broadcast(world, matrixB.row_ptr, 0);

  boost::mpi::broadcast(world, rowsRes, 0);
  boost::mpi::broadcast(world, colsRes, 0);
}

bool chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world.rank() == 0) {
    if (task_data->outputs.size() < 3) {
      throw std::runtime_error("There are not enough output buffers. Requires 3: values, col_indices, row_ptr");
    }

    auto* output_values = reinterpret_cast<double*>(task_data->outputs[0]);
    auto* output_col_indices = reinterpret_cast<int*>(task_data->outputs[1]);
    auto* output_row_ptr = reinterpret_cast<int*>(task_data->outputs[2]);

    std::ranges::copy(resultMatrix.values, output_values);
    std::ranges::copy(resultMatrix.col_indices, output_col_indices);
    std::ranges::copy(resultMatrix.row_ptr, output_row_ptr);
  }

  world.barrier();
  return true;
}
