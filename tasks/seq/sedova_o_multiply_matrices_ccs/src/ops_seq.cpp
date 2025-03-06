#include "seq/sedova_o_multiply_matrices_ccs/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool sedova_o_multiply_matrices_ccs_seq::TestTaskSequential::PreProcessingImpl() {
  rows_A_ = *reinterpret_cast<int*>(task_data->inputs[0]);
  cols_A_ = *reinterpret_cast<int*>(task_data->inputs[1]);
  rows_B_ = *reinterpret_cast<int*>(task_data->inputs[2]);
  cols_B_ = *reinterpret_cast<int*>(task_data->inputs[3]);

  // Загрузка матрицы A_
  auto* a_val_ptr = reinterpret_cast<double*>(task_data->inputs[4]);
  A_val_.assign(a_val_ptr, a_val_ptr + task_data->inputs_count[4]);

  auto* a_row_ind_ptr = reinterpret_cast<int*>(task_data->inputs[5]);
  A_row_ind_.assign(a_row_ind_ptr, a_row_ind_ptr + task_data->inputs_count[5]);

  auto* a_col_ptr_ptr = reinterpret_cast<int*>(task_data->inputs[6]);
  A_col_ptr_.assign(a_col_ptr_ptr, a_col_ptr_ptr + task_data->inputs_count[6]);

  // Загрузка матрицы B_
  auto* b_val_ptr = reinterpret_cast<double*>(task_data->inputs[7]);
  B_val_.assign(b_val_ptr, b_val_ptr + task_data->inputs_count[7]);

  auto* b_row_ind_ptr = reinterpret_cast<int*>(task_data->inputs[8]);
  B_row_ind_.assign(b_row_ind_ptr, b_row_ind_ptr + task_data->inputs_count[8]);

  auto* b_col_ptr_ptr = reinterpret_cast<int*>(task_data->inputs[9]);
  B_col_ptr_.assign(b_col_ptr_ptr, b_col_ptr_ptr + task_data->inputs_count[9]);

  // Транспонирование матрицы A_
  Transponirovanie(A_val_, A_row_ind_, A_col_ptr_, rows_A_, cols_A_, At_val_, At_row_ind_, At_col_ptr_);

  rows_At_ = cols_A_;
  cols_At_ = rows_A_;

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
  res_val_.clear();
  res_ind_.clear();
  res_ptr_.clear();

  res_ptr_.push_back(0);

  std::vector<int> x(rows_At_, -1);
  std::vector<double> x_values(rows_At_, 0.0);

  for (int col_b = 0; col_b < cols_B_; ++col_b) {
    std::ranges::fill(x.begin(), x.end(), -1);
    std::ranges::fill(x_values.begin(), x_values.end(), 0.0);

    for (int i = B_col_ptr_[col_b]; i < B_col_ptr_[col_b + 1]; ++i) {
      int row_b = B_row_ind_[i];
      x[row_b] = i;
      x_values[row_b] = B_val_[i];
    }

    for (int col_a = 0; col_a < static_cast<int>(At_col_ptr_.size() - 1); ++col_a) {
      double sum = 0.0;
      for (int i = At_col_ptr_[col_a]; i < At_col_ptr_[col_a + 1]; ++i) {
        int row_a = At_row_ind_[i];
        if (x[row_a] != -1) {
          sum += At_val_[i] * x_values[row_a];
        }
      }
      if (sum != 0.0) {
        res_val_.push_back(sum);
        res_ind_.push_back(col_a);
      }
    }

    res_ptr_.push_back(res_val_.size());  // NOLINT
  }

  return true;
}

bool sedova_o_multiply_matrices_ccs_seq::TestTaskSequential::PostProcessingImpl() {
  auto* c_val_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  auto* c_row_ind_ptr = reinterpret_cast<int*>(task_data->outputs[1]);
  auto* c_col_ptr_ptr = reinterpret_cast<int*>(task_data->outputs[2]);

  std::ranges::copy(res_val_.begin(), res_val_.end(), c_val_ptr);
  std::ranges::copy(res_ind_.begin(), res_ind_.end(), c_row_ind_ptr);
  std::ranges::copy(res_ptr_.begin(), res_ptr_.end(), c_col_ptr_ptr);

  return true;
}