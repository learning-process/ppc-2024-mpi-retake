#include "seq/chernova_n_matrix_multiplication_crs/include/ops_seq.hpp"

#include<iostream>
#include <cmath>
#include <cstddef>
#include <vector>

bool chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::PreProcessingImpl() {
  if (task_data->inputs.size() < 6 || task_data->inputs_count.size() < 6 || task_data->outputs.size() < 3) {
    throw std::runtime_error("Insufficient data in task_data");
  }

  auto* valuesA = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* colIndicesA = reinterpret_cast<int*>(task_data->inputs[1]);
  auto* rowPtrA = reinterpret_cast<int*>(task_data->inputs[2]);

  auto* valuesB = reinterpret_cast<double*>(task_data->inputs[3]);
  auto* colIndicesB = reinterpret_cast<int*>(task_data->inputs[4]);
  auto* rowPtrB = reinterpret_cast<int*>(task_data->inputs[5]);

  matrixA.values.assign(valuesA, valuesA + task_data->inputs_count[0]);
  matrixA.col_indices.assign(colIndicesA, colIndicesA + task_data->inputs_count[1]);
  matrixA.row_ptr.assign(rowPtrA, rowPtrA + task_data->inputs_count[2]);

  matrixB.values.assign(valuesB, valuesB + task_data->inputs_count[3]);
  matrixB.col_indices.assign(colIndicesB, colIndicesB + task_data->inputs_count[4]);
  matrixB.row_ptr.assign(rowPtrB, rowPtrB + task_data->inputs_count[5]);

  rowsA = static_cast<int>(task_data->inputs_count[2] - 1);
  rowsB = static_cast<int>(task_data->inputs_count[5] - 1);
  colsA = *std::max_element(matrixA.col_indices.begin(), matrixA.col_indices.end()) + 1;
  colsB = *std::max_element(matrixB.col_indices.begin(), matrixB.col_indices.end()) + 1;

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
      std::fill(temp.begin(), temp.end(), 0.0);
      int startA = matrixA.row_ptr[i];
      int endA = matrixA.row_ptr[i + 1];

      for (int posA = startA; posA < endA; ++posA) {
        double a_val = matrixA.values[posA];
        int a_col = matrixA.col_indices[posA];
        int startB = matrixB.row_ptr[a_col];
        int endB = matrixB.row_ptr[a_col + 1];

        for (int posB = startB; posB < endB; ++posB) {
          int b_col = matrixB.col_indices[posB];
          double b_val = matrixB.values[posB];
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
  auto* outputValues = reinterpret_cast<double*>(task_data->outputs[0]);
  auto* outputColIndices = reinterpret_cast<int*>(task_data->outputs[1]);
  auto* outputRowPtr = reinterpret_cast<int*>(task_data->outputs[2]);

  std::copy(resultMatrix.values.begin(), resultMatrix.values.end(), outputValues);
  std::copy(resultMatrix.col_indices.begin(), resultMatrix.col_indices.end(), outputColIndices);
  std::copy(resultMatrix.row_ptr.begin(), resultMatrix.row_ptr.end(), outputRowPtr);

  return true;
}