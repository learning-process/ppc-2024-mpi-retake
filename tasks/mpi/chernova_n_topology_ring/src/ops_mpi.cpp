#include "mpi/chernova_n_topology_ring/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool chernova_n_topology_ring_mpi::TestMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    vector_size_ = task_data->inputs_count[0];
    input_ = std::vector<char>(vector_size_);
    output_ = std::vector<char>(vector_size_);
    std::copy(task_data->inputs[0], task_data->inputs[0] + vector_size_, input_.data());
  }
  return true;
}

bool chernova_n_topology_ring_mpi::TestMPITaskParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->outputs_count.capacity() != 2 || task_data->inputs_count.size() != 1 ||
        task_data->inputs_count[0] <= 0) {
      return false;
    }
  }
  return true;
}

bool chernova_n_topology_ring_mpi::TestMPITaskParallel::RunImpl() {
  boost::mpi::broadcast(world_, vector_size_, 0);
  int world_rank = world_.rank();
  int world_size = world_.size();
  if (world_size != 1) {
    if (world_rank == 0) {
      process_.push_back(0);
      world_.send(world_rank + 1, 0, input_);
      world_.send(world_rank + 1, 0, process_);
    } else {
      std::vector<char> buffer(vector_size_);
      int tmp_recv = world_rank - 1;
      int tmp_send = (world_rank + 1) % world_size;

      world_.recv(tmp_recv, 0, buffer);
      world_.recv(tmp_recv, 0, process_);

      process_.push_back(world_rank);

      world_.send(tmp_send, 0, buffer);
      world_.send(tmp_send, 0, process_);
    }
    if (world_rank == 0) {
      world_.recv(world_.size() - 1, 0, output_);
      process_.resize(world_size);
      world_.recv(world_.size() - 1, 0, process_);
    }
  } else {
    process_.push_back(0);
    std::copy(input_.data(), input_.data() + vector_size_, output_.data());
  }
  return true;
}

bool chernova_n_topology_ring_mpi::TestMPITaskParallel::PostProcessingImpl() {
  world_.barrier();
  if (world_.rank() == 0) {
    for (int i = 0; i < vector_size_; ++i) {
      reinterpret_cast<char*>(task_data->outputs[0])[i] = output_[i];
    }
    for (int i = 0; i < world_.size(); ++i) {
      reinterpret_cast<int*>(task_data->outputs[1])[i] = process_[i];
    }
  }
  return true;
}