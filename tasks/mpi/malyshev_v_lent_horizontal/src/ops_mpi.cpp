#include "mpi/malyshev_v_lent_horizontal/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  //NOLINT
#include <vector>

namespace malyshev_v_lent_horizontal_mpi {

bool MatVecMultMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    matrix_ = std::vector<int>(task_data->inputs_count[0]);
    auto* tmp_matrix = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(tmp_matrix, tmp_matrix + task_data->inputs_count[0], matrix_.begin());

    vector_ = std::vector<int>(task_data->inputs_count[1]);
    auto* tmp_vector = reinterpret_cast<int*>(task_data->inputs[1]);
    std::copy(tmp_vector, tmp_vector + task_data->inputs_count[1], vector_.begin());

    rows_ = task_data->inputs_count[2];
    cols_ = task_data->inputs_count[3];
  }
  return true;
}

bool MatVecMultMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->inputs_count[2] == task_data->outputs_count[0]) &&
           (task_data->inputs_count[3] == task_data->inputs_count[1]);
  }
  return true;
}

bool MatVecMultMpi::RunImpl() {
  broadcast(world_, rows_, 0);
  broadcast(world_, cols_, 0);
  broadcast(world_, vector_, 0);

  const int world_size = world_.size();
  const int rows = static_cast<int>(rows_);

  int delta = rows / world_size;
  int last_row = rows % world_size;
  int local_n = (world_.rank() == world_size - 1) ? delta + last_row : delta;

  local_matrix_.resize(local_n * cols_);
  std::vector<int> send_counts(world_size);
  std::vector<int> recv_counts(world_size);

  for (int i = 0; i < world_size; ++i) {
    int count = (i == world_size - 1) ? delta + last_row : delta;
    send_counts[i] = count * static_cast<int>(cols_);
    recv_counts[i] = count;
  }

  boost::mpi::scatterv(world_, matrix_.data(), send_counts, local_matrix_.data(), 0);

  local_result_.resize(local_n);
  for (int i = 0; i < local_n; ++i) {
    local_result_[i] = 0;
    for (int j = 0; j < static_cast<int>(cols_); ++j) {
      local_result_[i] += local_matrix_[(i * cols_) + j] * vector_[j];
    }
  }

  std::vector<int> result(rows_);
  boost::mpi::gatherv(world_, local_result_.data(), static_cast<int>(local_result_.size()), result.data(), recv_counts,
                      0);

  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    std::ranges::copy(result.begin(), result.end(), output_ptr);
  }

  return true;
}

bool MatVecMultMpi::PostProcessingImpl() { return true; }

}  // namespace malyshev_v_lent_horizontal_mpi