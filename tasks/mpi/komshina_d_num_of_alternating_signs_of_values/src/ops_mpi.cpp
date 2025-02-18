#include "mpi/komshina_d_num_of_alternating_signs_of_values/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];
    auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  }
  result_ = 0;
  return true;
}

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] >= 2 && task_data->outputs_count[0] == 1;
  }
  return true;
}

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::RunImpl() {
  unsigned int chunk_size = 0, extra = 0;
  int rank = world_.rank(), size = world_.size();

  if (rank == 0) {
    chunk_size = task_data->inputs_count[0] / size;
    extra = task_data->inputs_count[0] % size;
  }

  boost::mpi::broadcast(world_, chunk_size, 0);
  boost::mpi::broadcast(world_, extra, 0);

  int local_size = chunk_size + (rank == size - 1 ? extra : 0);
  local_input_.resize(local_size);

  if (rank == 0) {
    for (int proc = 1; proc < size; ++proc) {
      int send_count = chunk_size + (proc == size - 1 ? extra : 0);
      world_.send(proc, 0, input_.data() + proc * chunk_size, send_count);
    }
    std::copy(input_.begin(), input_.begin() + local_size, local_input_.begin());
  } else {
    world_.recv(0, 0, local_input_.data(), local_size);
  }

  int local_count = 0;
  for (size_t i = 1; i < local_input_.size(); ++i) {
    if (local_input_[i - 1] * local_input_[i] < 0) {
      ++local_count;
    }
  }

  if (rank > 0) {
    int prev_value;
    world_.recv(rank - 1, 0, &prev_value, 1);
    if (!local_input_.empty() && (prev_value * local_input_[0] < 0)) {
      ++local_count;
    }
  }

  if (rank < size - 1) {
    int last_value = local_input_.empty() ? 0 : local_input_.back();
    world_.send(rank + 1, 0, &last_value, 1);
  }

  int total_count = 0;
  boost::mpi::reduce(world_, local_count, total_count, std::plus<>(), 0);

  if (rank == 0) {
    result_ = total_count;
  }

  return true;
}

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = result_;
  }
  return true;
}