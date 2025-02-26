#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
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
  int size_x = static_cast<int>(std::sqrt(world_.size()));
  int size_y = static_cast<int>(world_.size() / std::sqrt(world_.size()));
  if (size_x * size_y != world_size) {
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
  auto determine_next = [this]() {
    int dest_x = task_data_.target % size_x_;
    int dest_y = task_data_.target / size_x_;

    if (rank_x_ != dest_x) {
      return (rank_x_ < dest_x) ? right_ : left_;
    }
    if (rank_y_ != dest_y) {
      return (rank_y_ < dest_y) ? down_ : up_;
    }
    return -1;
  };

  if (world_.rank() == 0) {
    task_data_.path.push_back(0);
    int next_hop = determine_next();
    world_.send(next_hop, 0, task_data_.payload);
    world_.recv(boost::mpi::any_source, 0, task_data_.payload);

  } else {
    std::vector<char> received_data;
    world_.recv(boost::mpi::any_source, 0, received_data);

    task_data_.payload = received_data;
    task_data_.path.push_back(rank);

    if (rank != task_data_.target) {
      int next_hop = determine_next();
      world_.send(next_hop, 0, task_data_.payload);
    } else {
      world_.send(0, 0, task_data_.payload);
    }
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
