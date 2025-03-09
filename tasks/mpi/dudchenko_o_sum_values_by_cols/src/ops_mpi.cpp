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
    auto* tmp = reinterpret_cast<int*>(task_data->inputs[0]);
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

  int delta = static_cast<int>(cols_) / world_.size();
  int last_col = static_cast<int>(cols_) % world_.size();
  int local_n = (world_.rank() == world_.size() - 1) ? delta + last_col : delta;

  std::vector<int> column_major_input;
  if (world_.rank() == 0) {
    column_major_input.resize(rows_ * cols_);
    for (int j = 0; j < static_cast<int>(cols_); ++j) {
      for (int i = 0; i < static_cast<int>(rows_); ++i) {
        column_major_input[(j * rows_) + i] = input_[(i * cols_) + j];
      }
    }
  }

  std::vector<int> send_counts(world_.size());
  std::vector<int> displs(world_.size(), 0);

  for (int i = 0; i < world_.size(); ++i) {
    send_counts[i] = static_cast<int>(((i == world_.size() - 1) ? delta + last_col : delta) * rows_);
    if (i > 0) {
      displs[i] = displs[i - 1] + send_counts[i - 1];
    }
  }

  local_input_.resize(rows_ * local_n);

  boost::mpi::scatterv(world_, column_major_input.data(), send_counts, displs, local_input_.data(),
                       send_counts[world_.rank()], 0);

  std::vector<int> local_sum(local_n, 0);
  for (int j = 0; j < local_n; ++j) {
    for (int i = 0; i < static_cast<int>(rows_); ++i) {
      local_sum[j] += local_input_[(j * rows_) + i];
    }
  }

  std::vector<int> recv_counts(world_.size());
  for (int i = 0; i < world_.size(); ++i) {
    recv_counts[i] = (i == world_.size() - 1) ? delta + last_col : delta;
  }

  std::vector<int> displs_gath(world_.size(), 0);
  for (int i = 1; i < world_.size(); ++i) {
    displs_gath[i] = displs_gath[i - 1] + recv_counts[i - 1];
  }

  boost::mpi::gatherv(world_, local_sum.data(), static_cast<int>(local_sum.size()), sum_.data(), recv_counts,
                      displs_gath, 0);

  return true;
}

bool dudchenko_o_sum_values_by_cols_mpi::SumValByColsMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (unsigned int j = 0; j < cols_; j++) {
      reinterpret_cast<int*>(task_data->outputs[0])[j] = sum_[j];
    }
  }
  return true;
}
