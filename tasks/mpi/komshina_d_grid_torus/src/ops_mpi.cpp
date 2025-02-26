#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <utility>
#include <vector>

bool komshina_d_grid_torus_mpi::TestTaskMPI::PreProcessingImpl() {
  rank_x_ = world_.rank() % size_x_;
  rank_y_ = world_.rank() / size_x_;

  left_ = (rank_x_ - 1 + size_x_) % size_x_ + rank_y_ * size_x_;
  right_ = (rank_x_ + 1) % size_x_ + rank_y_ * size_x_;
  up_ = rank_x_ + ((rank_y_ - 1 + size_y_) % size_y_) * size_x_;
  down_ = rank_x_ + ((rank_y_ + 1) % size_y_) * size_x_;

  if (world_.rank() == 0) {
    auto* in_ptr = reinterpret_cast<TaskData*>(task_data->inputs[0]);
    task_data_ = *in_ptr;
    if (task_data_.target >= world_.size() || task_data_.target < 0) {
      return false;
    }
    task_data_.path.clear();
  }

  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::ValidationImpl() {
  int world_size = world_.size();
  int size_x_ = static_cast<int>(std::sqrt(world_.size()));
  int size_y_ = static_cast<int>(world_.size() / std::sqrt(world_.size()));
  if (size_x_ * size_y_ != world_size) {
    return false;
  }

  if (world_.rank() == 0) {
    if (task_data->inputs.empty() || task_data->inputs.size() != 1) {
      return false;
    }

    if (task_data->outputs.empty() || task_data->outputs_count.size() != 1) {
      return false;
    }
  }
  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::RunImpl() {
  int rank = world_.rank();
  int dest_x = task_data_.target % size_x_;
  int dest_y = task_data_.target / size_x_;

  auto determine_next = [this, dest_x, dest_y]() {
    if (rank_x_ != dest_x) {
      return (rank_x_ < dest_x) ? right_ : left_;
    }
    if (rank_y_ != dest_y) {
      return (rank_y_ < dest_y) ? down_ : up_;
    }
    return -1;
  };

  if (rank == 0) {
    task_data_.path.emplace_back(0);
    int next_hop = determine_next();
    world_.send(next_hop, 0, task_data_.payload);
    
    std::vector<char> buffer;
    world_.recv(boost::mpi::any_source, 0, buffer);

  } else {
    std::vector<char> buffer;
    world_.recv(boost::mpi::any_source, 0, buffer);

    task_data_.payload = std::move(buffer);
    task_data_.path.emplace_back(rank);

    int next_hop = (rank == task_data_.target) ? 0 : determine_next();
    world_.send(next_hop, 0, task_data_.payload);
  }

  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::PostProcessingImpl() {
  world_.barrier();
  if (world_.rank() == 0) {
    auto* out_ptr = reinterpret_cast<TaskData*>(task_data->outputs[0]);
    *out_ptr = task_data_;
  }
  return true;
}

std::vector<int> komshina_d_grid_torus_mpi::TestTaskMPI::CalculateRoute(int dest, int size_x, int size_y) {
  std::vector<int> path = {0};
  int current_x = 0;
  int current_y = 0;
  int dest_x = dest % size_x;
  int dest_y = dest / size_x;

  while (current_x != dest_x || current_y != dest_y) {
    if (current_x != dest_x) {
      current_x = (current_x + 1) % size_x;
    } else {
      current_y = (current_y + 1) % size_y;
    }
    path.push_back(current_x + (current_y * size_x));
  }
  return path;
}
