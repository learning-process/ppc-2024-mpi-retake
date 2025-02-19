#include "mpi/malyshev_v_lent_horizontal/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

bool malyshev_v_lent_horizontal::MatVecMultMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    // Init matrix and vector
    matrix_ = std::vector<int>(task_data->inputs_count[0]);
    auto *tmp_matrix = reinterpret_cast<int *>(task_data->inputs[0]);
    std::copy(tmp_matrix, tmp_matrix + task_data->inputs_count[0], matrix_.begin());

    vector_ = std::vector<int>(task_data->inputs_count[1]);
    auto *tmp_vector = reinterpret_cast<int *>(task_data->inputs[1]);
    std::copy(tmp_vector, tmp_vector + task_data->inputs_count[1], vector_.begin());

    rows_ = task_data->inputs_count[2];
    cols_ = task_data->inputs_count[3];
  }
  return true;
}

bool malyshev_v_lent_horizontal::MatVecMultMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->inputs_count[2] == task_data->outputs_count[0]) &&
           (task_data->inputs_count[3] == task_data->inputs_count[1]);
  }
  return true;
}

bool malyshev_v_lent_horizontal::MatVecMultMpi::RunImpl() {
  broadcast(world_, rows_, 0);
  broadcast(world_, cols_, 0);
  broadcast(world_, vector_, 0);

  int delta = (int)(rows_ / world_.size());
  int last_row = (int)(rows_ % world_.size());
  int local_n = (world_.rank() == world_.size() - 1) ? delta + last_row : delta;

  local_matrix_ = std::vector<int>(local_n * cols_);
  std::vector<int> send_counts(world_.size());
  std::vector<int> recv_counts(world_.size());
  for (int i = 0; i < world_.size(); ++i) {
    send_counts[i] = (i == world_.size() - 1) ? delta + last_row : delta;
    send_counts[i] *= (int)(cols_);
    recv_counts[i] = (i == world_.size() - 1) ? delta + last_row : delta;
  }
  boost::mpi::scatterv(world_, matrix_.data(), send_counts, local_matrix_.data(), 0);

  local_result_ = std::vector<int>(local_n, 0);
  for (int i = 0; i < local_n; ++i) {
    for (unsigned int j = 0; j < cols_; ++j) {
      local_result_[i] += local_matrix_[(i * cols_) + j] * vector_[j];
    }
  }

  std::vector<int> result(rows_, 0);
  boost::mpi::gatherv(world_, local_result_.data(), (int)local_result_.size(), result.data(), recv_counts, 0);

  if (world_.rank() == 0) {
    for (unsigned int i = 0; i < rows_; i++) {
      reinterpret_cast<int *>(task_data->outputs[0])[i] = result[i];
    }
  }

  return true;
}

bool malyshev_v_lent_horizontal::MatVecMultMpi::PostProcessingImpl() { return true; }