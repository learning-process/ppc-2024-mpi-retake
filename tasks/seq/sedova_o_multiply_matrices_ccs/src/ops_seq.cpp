#include "seq/sedova_o_multiply_matrices_ccs/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool sedova_o_multiply_matrices_ccs_seq::TestTaskSequential::PreProcessingImpl() {
  rowsA = *reinterpret_cast<int*>(task_data->inputs[0]);
  colsA = *reinterpret_cast<int*>(task_data->inputs[1]);
  rowsB = *reinterpret_cast<int*>(task_data->inputs[2]);
  colsB = *reinterpret_cast<int*>(task_data->inputs[3]);

  // Загрузка матрицы A
  auto* a_val_ptr = reinterpret_cast<double*>(task_data->inputs[4]);
  A_val.assign(a_val_ptr, a_val_ptr + task_data->inputs_count[4]);

  auto* a_row_ind_ptr = reinterpret_cast<int*>(task_data->inputs[5]);
  A_row_ind.assign(a_row_ind_ptr, a_row_ind_ptr + task_data->inputs_count[5]);

  auto* a_col_ptr_ptr = reinterpret_cast<int*>(task_data->inputs[6]);
  A_col_ptr.assign(a_col_ptr_ptr, a_col_ptr_ptr + task_data->inputs_count[6]);

  // Загрузка матрицы B
  auto* b_val_ptr = reinterpret_cast<double*>(task_data->inputs[7]);
  B_val.assign(b_val_ptr, b_val_ptr + task_data->inputs_count[7]);

  auto* b_row_ind_ptr = reinterpret_cast<int*>(task_data->inputs[8]);
  B_row_ind.assign(b_row_ind_ptr, b_row_ind_ptr + task_data->inputs_count[8]);

  auto* b_col_ptr_ptr = reinterpret_cast<int*>(task_data->inputs[9]);
  B_col_ptr.assign(b_col_ptr_ptr, b_col_ptr_ptr + task_data->inputs_count[9]);

  // Транспонирование матрицы A
  transpose_CCS(A_val, A_row_ind, A_col_ptr, rowsA, colsA, At_val, At_row_ind, At_col_ptr);

  rows_At = colsA;
  cols_At = rowsA;

  return true;
}

bool sedova_o_multiply_matrices_ccs_seq::TestTaskSequential::ValidationImpl() {
  int rows_a = *reinterpret_cast<int*>(task_data->inputs[0]);
  int cols_a = *reinterpret_cast<int*>(task_data->inputs[1]);
  int rows_b = *reinterpret_cast<int*>(task_data->inputs[2]);
  int cols_b = *reinterpret_cast<int*>(task_data->inputs[3]);

  return rows_a > 0 && cols_a > 0 && rows_b > 0 && cols_b > 0 && cols_a == rows_b;
}

bool sedova_o_multiply_matrices_ccs_seq::TestTaskSequential::RunImpl() {
  res_val.clear();
  res_ind.clear();
  res_ptr.clear();

  res_ptr.push_back(0);

  std::vector<int> x(rows_At, -1);
  std::vector<double> x_values(rows_At, 0.0);

  for (int col_b = 0; col_b < cols_B; ++col_b) {
    std::fill(x.begin(), x.end(), -1);
    std::fill(x_values.begin(), x_values.end(), 0.0);

    for (int i = B_col_ptr[col_b]; i < B_col_ptr[col_b + 1]; ++i) {
      int row_b = B_row_ind[i];
      x[row_b] = i;
      x_values[row_b] = B_val[i];
    }

    for (int col_a = 0; col_a < static_cast<int>(At_col_ptr.size() - 1); ++col_a) {
      double sum = 0.0;
      for (int i = At_col_ptr[col_a]; i < At_col_ptr[col_a + 1]; ++i) {
        int row_a = At_row_ind[i];
        if (x[row_a] != -1) {
          sum += At_val[i] * x_values[row_a];
        }
      }
      if (sum != 0.0) {
        res_val.push_back(sum);
        res_ind.push_back(col_a);
      }
    }

    res_ptr.push_back(res_val.size());
  }

  return true;
}

bool sedova_o_multiply_matrices_ccs_seq::TestTaskSequential::PostProcessingImpl() {
  auto* c_val_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  auto* c_row_ind_ptr = reinterpret_cast<int*>(task_data->outputs[1]);
  auto* c_col_ptr_ptr = reinterpret_cast<int*>(task_data->outputs[2]);

  std::copy(res_val.begin(), res_val.end(), c_val_ptr);
  std::copy(res_ind.begin(), res_ind.end(), c_row_ind_ptr);
  std::copy(res_ptr.begin(), res_ptr.end(), c_col_ptr_ptr);

  return true;
}