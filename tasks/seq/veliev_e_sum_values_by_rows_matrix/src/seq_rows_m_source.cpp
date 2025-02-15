#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "seq/veliev_e_sum_values_by_rows_matrix/include/seq_rows_m_header.hpp"
namespace veliev_e_sum_values_by_rows_matrix_seq {

void SeqProcForChecking(std::vector<int>& vec, int rows_size, std::vector<int>& output) {
  if (rows_size != 0) {
    int cnt = static_cast<int>(vec.size() / rows_size);
    output.resize(cnt);
    for (int i = 0; i < cnt; ++i) {
      output[i] = std::accumulate(vec.begin() + i * rows_size, vec.begin() + (i + 1) * rows_size, 0);
    }
  }
}

void GetRndMatrix(std::vector<int>& vec) {
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, static_cast<int>(vec.size()) - 1);
  std::ranges::generate(vec, [&dist, &reng] { return dist(reng); });
}

bool SumValuesByRowsMatrixSeq::PreProcessingImpl() {
  auto* ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  elem_total_ = ptr[0];
  rows_total_ = ptr[1];
  cols_total_ = ptr[2];

  input_ = std::vector<int>(elem_total_);
  std::ranges::copy(reinterpret_cast<int*>(task_data->inputs[1]),
                    reinterpret_cast<int*>(task_data->inputs[1]) + elem_total_, input_.begin());

  output_ = std::vector<int>(rows_total_);
  return true;
}

bool SumValuesByRowsMatrixSeq::RunImpl() {
  int row_sz = cols_total_;
  int original_rows_total = rows_total_;
  for (int i = 0; i < original_rows_total; ++i) {
    output_[i] = std::accumulate(input_.begin() + i * row_sz, input_.begin() + (i + 1) * row_sz, 0);
  }
  return true;
}

bool SumValuesByRowsMatrixSeq::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}

bool SumValuesByRowsMatrixSeq::ValidationImpl() {
  return task_data->inputs_count[0] == 3 && reinterpret_cast<int*>(task_data->inputs[0])[0] >= 0;
}
}  // namespace veliev_e_sum_values_by_rows_matrix_seq