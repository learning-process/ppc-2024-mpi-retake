#include "mpi/komshina_d_num_of_alternating_signs_of_values/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <cmath>
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
  unsigned int chunk_size = 0;
  unsigned int remainder = 0;

  if (world_.rank() == 0) {
    chunk_size = task_data->inputs_count[0] / world_.size();
    remainder = task_data->inputs_count[0] % world_.size();
  }

  broadcast(world_, chunk_size, 0);
  broadcast(world_, remainder, 0);

  int local_data_size = chunk_size + (world_.rank() == world_.size() - 1 ? remainder : 0);
  local_input_ = std::vector<int>(local_data_size);

  if (world_.rank() == 0) {
    std::copy(input_.begin(), input_.begin() + local_data_size, local_input_.begin());
  } else {
    world_.recv(0, 0, local_input_.data(), local_data_size);
  }

  int sign_changes = 0;
  for (size_t i = 1; i < local_input_.size(); ++i) {
    sign_changes += (local_input_[i - 1] * local_input_[i] < 0);
  }

  if (world_.rank() > 0) {
    int prev_value = 0;
    world_.recv(world_.rank() - 1, 0, &prev_value, 1);
    sign_changes += (prev_value * local_input_[0] < 0);
  }

  if (world_.rank() < world_.size() - 1) {
    int last_value = local_input_.back();
    world_.send(world_.rank() + 1, 0, &last_value, 1);
  }

  boost::mpi::reduce(world_, sign_changes, result_, std::plus<>(), 0);

  return true;
}

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = result_;
  }
  return true;
}