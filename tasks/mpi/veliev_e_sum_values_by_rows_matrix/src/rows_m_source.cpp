#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "mpi/veliev_e_sum_values_by_rows_matrix/include/rows_m_header.hpp"
namespace veliev_e_sum_values_by_rows_matrix_mpi {

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

bool SumValuesByRowsMatrixMpi::PreProcessingImpl() {
  int myid = world_.rank();

  if (myid == 0) {
    auto* ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    elem_total_ = ptr[0];
    rows_total_ = ptr[1];
    cols_total_ = ptr[2];

    input_ = std::vector<int>(elem_total_);
    std::ranges::copy(reinterpret_cast<int*>(task_data->inputs[1]),
                      reinterpret_cast<int*>(task_data->inputs[1]) + elem_total_, input_.begin());
    output_ = std::vector<int>(rows_total_);
  }
  return true;
}

bool SumValuesByRowsMatrixMpi::RunImpl() {
  int myid = world_.rank();
  int world_size = world_.size();
  int original_rows_total = rows_total_;
  int row_sz = 0;
  int rows_for_each = 0;
  int remainder = 0;

  // for 1 proc run
  if (world_size == 1) {
    output_.resize(rows_total_);
    for (int i = 0; i < rows_total_; ++i) {
      output_[i] = std::accumulate(input_.begin() + i * cols_total_, input_.begin() + (i + 1) * cols_total_, 0);
    }
    return true;
  }

  if (world_.rank() == 0) {
    original_rows_total = rows_total_;
    rows_for_each = rows_total_ / world_size;
    remainder = rows_total_ % world_size;
    row_sz = cols_total_;
    if (remainder != 0) {
      rows_total_ += (world_size - remainder);
      input_.resize(rows_total_ * row_sz, 0);
      rows_for_each = rows_total_ / world_size;
    }
  }
  broadcast(world_, row_sz, 0);
  broadcast(world_, rows_for_each, 0);

  std::vector<int> loc_vec(row_sz * rows_for_each);
  scatter(world_, myid == 0 ? input_.data() : nullptr, loc_vec.data(), row_sz * rows_for_each, 0);
  std::vector<int> local_sums(rows_for_each, 0);

  for (int i = 0; i < rows_for_each; ++i) {
    local_sums[i] = std::accumulate(loc_vec.begin() + i * row_sz, loc_vec.begin() + (i + 1) * row_sz, 0);
  }

  if (world_.rank() == 0) {
    output_.resize(rows_total_);
  }

  gather(world_, local_sums.data(), rows_for_each, output_.data(), 0);

  if (world_.rank() == 0) {
    output_.resize(original_rows_total);
  }

  return true;
}

bool SumValuesByRowsMatrixMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(output_, reinterpret_cast<int*>(task_data->outputs[0]));
  }
  return true;
}

bool SumValuesByRowsMatrixMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == 3 && reinterpret_cast<int*>(task_data->inputs[0])[0] >= 0;
  }
  return true;
}
}  // namespace veliev_e_sum_values_by_rows_matrix_mpi