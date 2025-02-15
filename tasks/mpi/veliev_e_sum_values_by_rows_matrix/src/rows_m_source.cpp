#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "mpi/veliev_e_sum_values_by_rows_matrix/include/rows_m_header.hpp"
namespace veliev_e_sum_values_by_rows_matrix_mpi {

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

bool SumValuesByRowsMatrixMpi::PreProcessingImpl() {
  int myid = world_.rank();

  if (myid == 0) {
    auto* ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    elem_total_ = ptr[0];
    rows_total_ = ptr[1];
    cols_total_ = ptr[2];

    input_ = std::vector<int>(elem_total_);
    void* ptr_r = task_data->inputs[1];
    void* ptr_d = input_.data();
    memcpy(ptr_d, ptr_r, sizeof(int) * elem_total_);
    output_ = std::vector<int>(rows_total_);
  }
  return true;
}

bool SumValuesByRowsMatrixMpi::RunImpl() {
  int myid = world_.rank();
  int world_size = world_.size();
  int row_sz = cols_total_;
  int original_rows_total_ = rows_total_;
  int rows_for_each = rows_total_ / world_size;
  int remainder = rows_total_ % world_size;
  if (myid == 0) {
    if (remainder != 0) {
      rows_total_ += (world_size - remainder);
      input_.resize(rows_total_ * row_sz, 0);
      rows_for_each = rows_total_ / world_size;
    }
  }

  if (world_size == 1) {
    output_.resize(original_rows_total_);
    for (int i = 0; i < original_rows_total_; ++i) {
      output_[i] = std::accumulate(input_.begin() + i * row_sz, input_.begin() + (i + 1) * row_sz, 0);
    }
    return true;
  }
  boost::mpi::broadcast(world_, row_sz, 0);
  boost::mpi::broadcast(world_, rows_for_each, 0);

  std::vector<int> loc_vec(row_sz * rows_for_each);
  boost::mpi::scatter(world_, myid == 0 ? input_.data() : nullptr, loc_vec.data(), row_sz * rows_for_each, 0);

  std::vector<int> local_sums(rows_for_each, 0);
  for (int i = 0; i < rows_for_each; ++i) {
    local_sums[i] = std::accumulate(loc_vec.begin() + i * row_sz, loc_vec.begin() + (i + 1) * row_sz, 0);
  }

  if (myid == 0) {
    output_.resize(rows_total_);
  }

  boost::mpi::gather(world_, local_sums.data(), rows_for_each, output_.data(), 0);

  if (myid == 0) {
    output_.resize(original_rows_total_);
  }

  return true;
}

bool SumValuesByRowsMatrixMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::copy(output_.begin(), output_.end(), reinterpret_cast<int*>(task_data->outputs[0]));
  }
  return true;
}

bool SumValuesByRowsMatrixMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs_count[0] != 3 || reinterpret_cast<int*>(task_data->inputs[0])[0] < 0) {
      return false;
    }
  }

  return true;
}
}  // namespace veliev_e_sum_values_by_rows_matrix_mpi