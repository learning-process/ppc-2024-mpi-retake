#include "mpi/khokhlov_a_sum_values_by_rows/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

bool khokhlov_a_sum_values_by_rows_mpi::SumValByRowsMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(task_data->inputs_count[0]);
    auto *tmp = reinterpret_cast<int *>(task_data->inputs[0]);
    std::copy(tmp, tmp + task_data->inputs_count[0], input_.begin());
    row_ = task_data->inputs_count[1];
    col_ = task_data->inputs_count[2];
    // Init value for output
    sum_ = std::vector<int>(row_, 0);
  }
  return true;
}

bool khokhlov_a_sum_values_by_rows_mpi::SumValByRowsMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->inputs_count[1] == task_data->outputs_count[0]);
  }
  return true;
}

bool khokhlov_a_sum_values_by_rows_mpi::SumValByRowsMpi::RunImpl() {
  broadcast(world_, row_, 0);
  broadcast(world_, col_, 0);

  int delta = (int)(row_ / world_.size());
  int last_row = (int)(row_ % world_.size());
  int local_n = (world_.rank() == world_.size() - 1) ? delta + last_row : delta;

  local_input_ = std::vector<int>(local_n * col_);
  std::vector<int> send_counts(world_.size());
  std::vector<int> recv_counts(world_.size());
  for (int i = 0; i < world_.size(); ++i) {
    send_counts[i] = (i == world_.size() - 1) ? delta + last_row : delta;
    send_counts[i] *= (int)(col_);
    recv_counts[i] = (i == world_.size() - 1) ? delta + last_row : delta;
  }
  boost::mpi::scatterv(world_, input_.data(), send_counts, local_input_.data(), 0);

  std::vector<int> local_sum(local_n, 0);
  for (int i = 0; i < local_n; ++i) {
    for (unsigned int j = 0; j < col_; ++j) {
      local_sum[i] += local_input_[(i * col_) + j];
    }
  }

  boost::mpi::gatherv(world_, local_sum.data(), (int)local_sum.size(), sum_.data(), recv_counts, 0);

  return true;
}

bool khokhlov_a_sum_values_by_rows_mpi::SumValByRowsMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (unsigned int i = 0; i < row_; i++) {
      reinterpret_cast<int *>(task_data->outputs[0])[i] = sum_[i];
    }
  }
  return true;
}
