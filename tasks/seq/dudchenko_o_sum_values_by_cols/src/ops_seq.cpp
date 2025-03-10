#include <algorithm>
#include <vector>

#include "seq/dudchenko_o_sum_values_by_cols/include/ops_sec.hpp"

using namespace std::chrono_literals;

bool dudchenko_o_sum_values_by_cols_seq::SumValByCols::PreProcessingImpl() {
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto *tmp = reinterpret_cast<int *>(task_data->inputs[0]);
  std::copy(tmp, tmp + task_data->inputs_count[0], input_.begin());
  rows_ = task_data->inputs_count[1];
  cols_ = task_data->inputs_count[2];
  sum_ = std::vector<int>(cols_, 0);
  return true;
}

bool dudchenko_o_sum_values_by_cols_seq::SumValByCols::ValidationImpl() {
  return (task_data->inputs_count[2] == task_data->outputs_count[0]);
}

bool dudchenko_o_sum_values_by_cols_seq::SumValByCols::RunImpl() {
  for (unsigned int j = 0; j < cols_; j++) {
    int tmp_sum = 0;
    for (unsigned int i = 0; i < rows_; i++) {
      tmp_sum += input_[(i * cols_) + j];
    }
    sum_[j] = tmp_sum;
  }
  return true;
}

bool dudchenko_o_sum_values_by_cols_seq::SumValByCols::PostProcessingImpl() {
  for (unsigned int j = 0; j < cols_; j++) {
    reinterpret_cast<int *>(task_data->outputs[0])[j] = sum_[j];
  }
  return true;
}
