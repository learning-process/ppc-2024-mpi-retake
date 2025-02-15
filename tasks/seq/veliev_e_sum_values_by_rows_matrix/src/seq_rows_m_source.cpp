#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "seq/veliev_e_sum_values_by_rows_matrix/include/seq_rows_m_header.hpp"
namespace veliev_e_sum_values_by_rows_matrix_seq {

void SeqProcForChecking(std::vector<int>& arr, int row_sz, std::vector<int>& output) {
  if (row_sz != 0) {
    int cnt = arr.size() / row_sz;
    output.resize(cnt);
    for (int i = 0; i < cnt; ++i) {
      output[i] = std::accumulate(arr.begin() + i * row_sz, arr.begin() + (i + 1) * row_sz, 0);
    }
  }
}

void GetRndMatrix(std::vector<int>& vec) {
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, vec.size());
  std::generate(vec.begin(), vec.end(), [&dist, &reng] { return dist(reng); });
}

bool SumValuesByRowsMatrixSeq::PreProcessingImpl() {
  auto* ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  elem_total_ = ptr[0];
  rows_total_ = ptr[1];
  cols_total_ = ptr[2];

  input_ = std::vector<int>(elem_total_);
  void* ptr_r = task_data->inputs[1];
  void* ptr_d = input_.data();
  memcpy(ptr_d, ptr_r, sizeof(int) * elem_total_);
  output_ = std::vector<int>(rows_total_);
  return true;
}

bool SumValuesByRowsMatrixSeq::RunImpl() {
  int row_sz = cols_total_;
  int original_rows_total_ = rows_total_;
  for (int i = 0; i < original_rows_total_; ++i) {
    output_[i] = std::accumulate(input_.begin() + i * row_sz, input_.begin() + (i + 1) * row_sz, 0);
  }
  return true;
}

bool SumValuesByRowsMatrixSeq::PostProcessingImpl() {
  std::copy(output_.begin(), output_.end(), reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}

bool SumValuesByRowsMatrixSeq::ValidationImpl() {
  if (task_data->inputs_count[0] != 3 || reinterpret_cast<int*>(task_data->inputs[0])[0] < 0) {
    return false;
  }

  return true;
}
}  // namespace veliev_e_sum_values_by_rows_matrix_seq