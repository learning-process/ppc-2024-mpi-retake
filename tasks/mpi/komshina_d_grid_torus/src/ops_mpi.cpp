#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <utility>
#include <vector>
#include <ranges>


bool komshina_d_grid_torus_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto* task_data_ptr = reinterpret_cast<TaskData*>(task_data->inputs[0]);
    task_data_ = *task_data_ptr;
    if (task_data_.target < 0 || task_data_.target >= world_.size()) {
      return false;
    }
    task_data_.path.clear();
  }
  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::ValidationImpl() {
  if (world_.size() % width != 0) {
    return false;
  }
  if (world_.rank() == 0) {
    if (task_data->inputs.empty() || task_data->inputs[0] == nullptr) {
      return false;
    }
    if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) {
      return false;
    }
  }
  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::RunImpl() {
  int current_rank = world_.rank();

  if (current_rank == 0) {
    task_data_.path.push_back(0);
    int next = GetNextHop(0, task_data_.target, width, height);
    world_.send(next, 0, task_data_);
    world_.recv(boost::mpi::any_source, 0, task_data_);
  } else {
    world_.recv(boost::mpi::any_source, 0, task_data_);
    if (task_data_.path[0] == -1) {
      return true;
    }
    task_data_.path.push_back(world_.rank());
    if (current_rank != task_data_.target) {
      int next = GetNextHop(current_rank, task_data_.target, width, height);
      world_.send(next, 0, task_data_);
    } else {
      world_.send(0, 0, task_data_);
    }
  }
  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::PostProcessingImpl() {
  world_.barrier();
  if (world_.rank() == 0) {
    auto* output_data_ptr = reinterpret_cast<TaskData*>(task_data->outputs[0]);
    *output_data_ptr = task_data_;
  }
  return true;
}

namespace komshina_d_grid_torus_mpi {

void komshina_d_grid_torus_mpi::TestTaskMPI::ComputeGridSize() {
  int world_size = world_.size();
  height = static_cast<int>(std::sqrt(world_size));
  width = world_size / height;
}

int komshina_d_grid_torus_mpi::TestTaskMPI::GetNextHop(int current, int target, int width, int height) {
  int current_x = current % width;
  int current_y = current / width;
  int target_x = target % width;
  int target_y = target / width;

  if (current_x != target_x) {
    return (current_x < target_x) ? current + 1 : current - 1;
  } else {
    return (current_y < target_y) ? current + width : current - width;
  }
}

std::vector<int> komshina_d_grid_torus_mpi::TestTaskMPI::ComputePath(int target, int world_size, int width,
                                                                        int height) {
  std::vector<int> path = {0};
  int current = 0;
  while (current != target) {
    current = GetNextHop(current, target, width, height);
    path.push_back(current);
  }
  return path;
}

}  // namespace komshina_d_grid_torus_mpi
