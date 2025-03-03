#include "mpi/dudchenko_o_sum_values_by_cols/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

bool dudchenko_o_sum_values_by_cols_mpi::SumValByColsMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    input_ = std::vector<int>(task_data->inputs_count[0]);
    auto *tmp = reinterpret_cast<int *>(task_data->inputs[0]);
    std::copy(tmp, tmp + task_data->inputs_count[0], input_.begin());
    rows_ = task_data->inputs_count[1];
    cols_ = task_data->inputs_count[2];
    sum_ = std::vector<int>(cols_, 0);
  }
  return true;
}

bool dudchenko_o_sum_values_by_cols_mpi::SumValByColsMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->inputs_count[2] == task_data->outputs_count[0]);
  }
  return true;
}

bool dudchenko_o_sum_values_by_cols_mpi::SumValByColsMpi::RunImpl() {
  broadcast(world_, rows_, 0);
  broadcast(world_, cols_, 0);

  int delta = (int)(cols_ / world_.size());
  int last_col = (int)(cols_ % world_.size());
  int local_n = (world_.rank() == world_.size() - 1) ? delta + last_col : delta;

  local_input_ = std::vector<int>(rows_ * local_n);
  std::vector<int> send_counts(world_.size());
  std::vector<int> recv_counts(world_.size());
  for (int i = 0; i < world_.size(); ++i) {
    send_counts[i] = (i == world_.size() - 1) ? delta + last_col : delta;
    send_counts[i] *= (int)(rows_);
    recv_counts[i] = (i == world_.size() - 1) ? delta + last_col : delta;
  }
  boost::mpi::scatterv(world_, input_.data(), send_counts, local_input_.data(), 0);

  std::vector<int> local_sum(local_n, 0);
  for (int j = 0; j < local_n; ++j) {
    for (unsigned int i = 0; i < rows_; ++i) {
      local_sum[j] += local_input_[(i * local_n) + j];
    }
  }

  boost::mpi::gatherv(world_, local_sum.data(), (int)local_sum.size(), sum_.data(), recv_counts, 0);

  return true;
}

bool dudchenko_o_sum_values_by_cols_mpi::SumValByColsMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (unsigned int j = 0; j < cols_; j++) {
      reinterpret_cast<int *>(task_data->outputs[0])[j] = sum_[j];
    }
  }
  return true;
}
