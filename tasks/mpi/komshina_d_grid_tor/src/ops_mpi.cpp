#include "mpi/komshina_d_grid_tor/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/exception.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace std::chrono_literals;

bool komshina_d_grid_torus_topology_mpi::TestTaskMPI::PreProcessingImpl() { return ValidationImpl(); }

bool komshina_d_grid_torus_topology_mpi::TestTaskMPI::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->inputs_count.empty()) {
    return false;
  }

  for (size_t i = 0; i < task_data->inputs.size(); ++i) {
    if (task_data->inputs_count[i] <= 0 || task_data->inputs[i] == nullptr) {
      return false;
    }
  }

  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }

  int size = boost::mpi::communicator().size();
  int sqrt_size = static_cast<int>(std::sqrt(size));
  return sqrt_size * sqrt_size == size;

  return true;
}

bool komshina_d_grid_torus_topology_mpi::TestTaskMPI::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();
  int grid_size = static_cast<int>(std::sqrt(size));

  world_.barrier();

  for (int step = 0; step < grid_size; ++step) {
    auto neighbors = ComputeNeighbors(rank, grid_size);

    for (int neighbor : neighbors) {
      if (neighbor < size) {
        std::vector<uint8_t> send_data(task_data->inputs_count[0], 0);
        std::copy(task_data->inputs[0], task_data->inputs[0] + task_data->inputs_count[0], send_data.begin());

        try {
          world_.send(neighbor, 0, send_data);
          std::vector<uint8_t> recv_data(task_data->inputs_count[0]);
          world_.recv(neighbor, 0, recv_data);

          if (task_data->outputs_count[0] >= send_data.size()) {
            std::ranges::copy(send_data, task_data->outputs[0]);
          }
        } catch (const boost::mpi::exception& e) {
          std::cerr << "Error when exchanging data with process " << neighbor << ": " << e.what() << "\n";
        }
      }
    }
    world_.barrier();
  }
  return true;
}

bool komshina_d_grid_torus_topology_mpi::TestTaskMPI::PostProcessingImpl() { return true; }

std::vector<int> komshina_d_grid_torus_topology_mpi::TestTaskMPI::ComputeNeighbors(int rank, int grid_size) {
  int x = rank % grid_size;
  int y = rank / grid_size;

  int left = ((x - 1 + grid_size) % grid_size) + (y * grid_size);
  int right = ((x + 1) % grid_size) + (y * grid_size);

  int up = x + (((y - 1 + grid_size) % grid_size) * grid_size);
  int down = x + (((y + 1) % grid_size) * grid_size);

  return {left, right, up, down};
}