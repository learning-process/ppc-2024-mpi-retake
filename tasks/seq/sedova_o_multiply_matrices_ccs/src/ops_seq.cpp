#include "seq/sedova_o_multiply_matrices_ccs/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool sedova_o_multiply_matrices_ccs_seq::TestTaskSequential::PreProcessingImpl() {
  rows_A = *reinterpret_cast<int *>(taskData->inputs[0]);
  cols_A = *reinterpret_cast<int *>(taskData->inputs[1]);
  rows_B = *reinterpret_cast<int *>(taskData->inputs[2]);
  cols_B = *reinterpret_cast<int *>(taskData->inputs[3]);

  // Загрузка матрицы A
  auto AValPtr = reinterpret_cast<double *>(taskData->inputs[4]);
  A_val.assign(AValPtr, AValPtr + taskData->inputs_count[4]);

  auto ARowIndPtr = reinterpret_cast<int *>(taskData->inputs[5]);
  A_row_ind.assign(ARowIndPtr, ARowIndPtr + taskData->inputs_count[5]);

  auto AColPtrPtr = reinterpret_cast<int *>(taskData->inputs[6]);
  A_col_ptr.assign(AColPtrPtr, AColPtrPtr + taskData->inputs_count[6]);

  // Загрузка матрицы B
  auto BValPtr = reinterpret_cast<double *>(taskData->inputs[7]);
  B_val.assign(BValPtr, BValPtr + taskData->inputs_count[7]);

  auto BRowIndPtr = reinterpret_cast<int *>(taskData->inputs[8]);
  B_row_ind.assign(BRowIndPtr, BRowIndPtr + taskData->inputs_count[8]);

  auto BColPtrPtr = reinterpret_cast<int *>(taskData->inputs[9]);
  B_col_ptr.assign(BColPtrPtr, BColPtrPtr + taskData->inputs_count[9]);

  // Транспонирование матрицы A
  Transponirovanie(A_val, A_row_ind, A_col_ptr, rows_A, cols_A, At_val, At_row_ind, At_col_ptr);

  rows_At = cols_A;
  cols_At = rows_A;

  return true;
}

bool sedova_o_multiply_matrices_ccs_seq::TestTaskSequential::ValidationImpl() {
  int rowsA = *reinterpret_cast<int *>(taskData->inputs[0]);
  int colsA = *reinterpret_cast<int *>(taskData->inputs[1]);
  int rowsB = *reinterpret_cast<int *>(taskData->inputs[2]);
  int colsB = *reinterpret_cast<int *>(taskData->inputs[3]);

  return rowsA > 0 && colsA > 0 && rowsB > 0 && colsB > 0 && colsA == rowsB;
}


bool sedova_o_multiply_matrices_ccs_seq::TestTaskSequential::RunImpl() {
  res_val.clear();
  res_ind.clear();
  res_ptr.clear();

  res_ptr.push_back(0);

  std::vector<int> X(rows_At, -1);
  std::vector<double> XValues(rows_At, 0.0);

  for (int colB = 0; colB < cols_B; ++colB) {
    std::fill(X.begin(), X.end(), -1);
    std::fill(XValues.begin(), XValues.end(), 0.0);

    for (int i = B_col_ptr[colB]; i < B_col_ptr[colB + 1]; ++i) {
      int rowB = B_row_ind[i];
      X[rowB] = i;
      XValues[rowB] = B_val[i];
    }

    for (int colA = 0; colA < static_cast<int>(At_col_ptr.size() - 1); ++colA) {
      double sum = 0.0;
      for (int i = At_col_ptr[colA]; i < At_col_ptr[colA + 1]; ++i) {
        int rowA = At_row_ind[i];
        if (X[rowA] != -1) {
          sum += At_val[i] * XValues[rowA];
        }
      }
      if (sum != 0.0) {
        res_val.push_back(sum);
        res_ind.push_back(colA);
      }
    }

    res_ptr.push_back(res_val.size());
  }
}

bool sedova_o_multiply_matrices_ccs_seq::TestTaskSequential::PostProcessingImpl() {
  auto CValPtr = reinterpret_cast<double *>(taskData->outputs[0]);
  auto CRowIndPtr = reinterpret_cast<int *>(taskData->outputs[1]);
  auto CColPtrPtr = reinterpret_cast<int *>(taskData->outputs[2]);

  std::copy(res_val.begin(), res_val.end(), CValPtr);
  std::copy(res_ind.begin(), res_ind.end(), CRowIndPtr);
  std::copy(res_ptr.begin(), res_ptr.end(), CColPtrPtr);

  return true;
}