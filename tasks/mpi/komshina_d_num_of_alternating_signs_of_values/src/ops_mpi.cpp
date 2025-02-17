#include "mpi/komshina_d_num_of_alternating_signs_of_values/include/ops_mpi.hpp"

#include <cmath>
#include <cstddef>
#include <vector>
#include <algorithm>

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
  int world_size = world_.size();
  int world_rank = world_.rank();

  int local_size = task_data->inputs_count[0] / world_size;
  int remainder = task_data->inputs_count[0] % world_size;
  if (world_rank < remainder) {
    local_size++;
  }

  std::vector<int> local_input(local_size);

  if (world_rank == 0) {
    std::vector<int> send_counts(world_size, task_data->inputs_count[0] / world_size);
    std::vector<int> displacements(world_size, 0);

    for (int i = 0; i < remainder; ++i) {
      send_counts[i]++;
    }
    for (int i = 1; i < world_size; ++i) {
      displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }

    std::copy(input_.begin(), input_.begin() + local_size, local_input.begin());

    for (int i = 1; i < world_size; ++i) {
      world_.send(i, 0, input_.data() + displacements[i], send_counts[i]);
    }
  } else {
    world_.recv(0, 0, local_input.data(), local_size);
  }

  int local_count = 0;
  if (local_input.size() > 1) {
    for (size_t i = 1; i < local_input.size(); ++i) {
      if ((input_[i] * input_[i - 1]) < 0) {
        local_count++;
      }
    }
  }

  int global_count = 0;
  boost::mpi::reduce(world_, local_count, global_count, std::plus<>(), 0);

  if (world_rank == 0) {
    result_ = global_count;
  }

  return true;
}

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = result_;
  }
  return true;
}