#include "mpi/komshina_d_num_of_alternating_signs_of_values/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto input_size = static_cast<std::size_t>(task_data->inputs_count[0]);
    auto *data_ptr = reinterpret_cast<int32_t *>(task_data->inputs[0]);
    input_.assign(data_ptr, data_ptr + input_size);
  }
  result_ = 0;
  return true;
}

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::ValidationImpl() {
  return world_.rank() != 0 || (task_data->inputs_count[0] >= 2 && task_data->outputs_count[0] == 1);
}

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::RunImpl() {
  std::size_t chunk_size = 0;
  std::size_t remainder = 0;

  if (world_.rank() == 0) {
    chunk_size = static_cast<std::size_t>(task_data->inputs_count[0]) / static_cast<std::size_t>(world_.size());
    remainder = static_cast<std::size_t>(task_data->inputs_count[0]) % static_cast<std::size_t>(world_.size());
  }

  if (world_.rank() == 0) {
    for (int proc = 1; proc < world_.size(); ++proc) {
      world_.send(proc, 0, chunk_size);
      world_.send(proc, 0, remainder);
    }
  } else {
    world_.recv(0, 0, chunk_size);
    world_.recv(0, 0, remainder);
  }

  std::size_t local_size = chunk_size + (world_.rank() == world_.size() - 1 ? remainder : 0);
  local_input_.resize(local_size);

  if (world_.rank() == 0) {
    for (int proc = 1; proc < world_.size(); ++proc) {
      std::size_t offset = proc * chunk_size;
      std::size_t send_size = chunk_size + (proc == world_.size() - 1 ? remainder : 0);
      world_.send(proc, 0, input_.data() + static_cast<std::ptrdiff_t>(offset), static_cast<int>(send_size));
    }
    std::copy(input_.begin(), input_.begin() + static_cast<std::ptrdiff_t>(local_size), local_input_.begin());
  } else {
    world_.recv(0, 0, local_input_.data(), static_cast<int>(local_size));
  }

  int32_t local_count = 0;
  for (std::size_t i = 1; i < local_input_.size(); ++i) {
    if (local_input_[i - 1] * local_input_[i] < 0) {
      ++local_count;
    }
  }

  int32_t prev_value = 0;
  if (world_.rank() > 0) {
    world_.recv(world_.rank() - 1, 0, &prev_value, 1);
    if (!local_input_.empty() && prev_value * local_input_[0] < 0) {
      ++local_count;
    }
  }

  int32_t last_value = local_input_.empty() ? 0 : local_input_.back();
  if (world_.rank() < world_.size() - 1) {
    world_.send(world_.rank() + 1, 0, &last_value, 1);
  }

  if (world_.rank() == 0) {
    result_ = local_count;
    for (int proc = 1; proc < world_.size(); ++proc) {
      int32_t recv_count = 0;
      world_.recv(proc, 0, &recv_count, 1);
      result_ += recv_count;
    }
  } else {
    world_.send(0, 0, &local_count, 1);
  }

  return true;
}

bool komshina_d_num_of_alternations_signs_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int32_t *>(task_data->outputs[0])[0] = result_;
  }
  return true;
}