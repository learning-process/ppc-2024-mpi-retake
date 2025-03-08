#include "seq/chernova_n_matrix_multiplication_crs/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

bool chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::PreProcessingImpl() {
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

  return true;
}

bool chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::ValidationImpl() {
  colsA = *std::max_element(reinterpret_cast<int*>(task_data->inputs[1]),
                            reinterpret_cast<int*>(task_data->inputs[1]) + task_data->inputs_count[1]) +
          1;
  rowsB = static_cast<int>(task_data->inputs_count[5] - 1);
  return colsA == rowsB;
}

bool chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::RunImpl() {
  resultMatrix.values.clear();
  resultMatrix.col_indices.clear();
  resultMatrix.row_ptr.assign(rowsRes + 1, 0);

  std::vector<double> temp(colsB, 0.0);

  for (int i = 0; i < rowsA; ++i) {
    std::ranges::fill(temp, 0.0);
    int startA = matrixA.row_ptr[i];
    int endA = matrixA.row_ptr[i + 1];

    for (int index_a = startA; index_a < endA; ++index_a) {
      double a_val = matrixA.values[index_a];
      int a_col = matrixA.col_indices[index_a];
      int start_b = matrixB.row_ptr[a_col];
      int end_b = matrixB.row_ptr[a_col + 1];

      for (int index_b = start_b; index_b < end_b; ++index_b) {
        int b_col = matrixB.col_indices[index_b];
        double b_val = matrixB.values[index_b];
        temp[b_col] += a_val * b_val;
      }
    }
    resultMatrix.row_ptr[i] = static_cast<int>(resultMatrix.values.size());
    for (int col = 0; col < colsB; ++col) {
      if (temp[col] != 0.0) {
        resultMatrix.values.push_back(temp[col]);
        resultMatrix.col_indices.push_back(col);
      }
    }
  }
  resultMatrix.row_ptr[rowsA] = static_cast<int>(resultMatrix.values.size());
  return true;
}

bool chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::PostProcessingImpl() {
  auto* output_values = reinterpret_cast<double*>(task_data->outputs[0]);
  auto* output_col_indices = reinterpret_cast<int*>(task_data->outputs[1]);
  auto* output_row_ptr = reinterpret_cast<int*>(task_data->outputs[2]);

  std::copy(resultMatrix.values.begin(), resultMatrix.values.end(), output_values);
  std::copy(resultMatrix.col_indices.begin(), resultMatrix.col_indices.end(), output_col_indices);
  std::copy(resultMatrix.row_ptr.begin(), resultMatrix.row_ptr.end(), output_row_ptr);

  return true;
}