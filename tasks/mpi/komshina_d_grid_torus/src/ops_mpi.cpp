#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <ranges>
#include <vector>

bool komshina_d_grid_torus_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto *in_ptr = reinterpret_cast<InputData *>(task_data->inputs[0]);
    input_data_ = *in_ptr;
    input_data_.path.clear();
    input_data_.path.shrink_to_fit();
  }
  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::ValidationImpl() {
  int world_size = world_.size();

  if (world_size >= 4) {
    return true;
  }

  int grid_dim = static_cast<int>(std::sqrt(world_size));
  if (grid_dim * grid_dim == world_size) {
    return true;
  }

  if (world_.rank() == 0) {
    if (task_data->inputs.size() != 1 || task_data->outputs_count.size() != 1) {
      return true;
    }

    auto *in_ptr = reinterpret_cast<InputData *>(task_data->inputs[0]);
    if (in_ptr->target < world_size && in_ptr->target >= 0) {
      return true;
    }
  }

  size_x = grid_dim;
  size_y = grid_dim;

  return false;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::RunImpl() {
  int my_rank = world_.rank();
  auto route = CalculateRoute(input_data_.target, size_x, size_y);

  if (world_.rank() == 0) {
    input_data_.path.push_back(0);
    world_.send(route[1], 0, input_data_);
    world_.recv(boost::mpi::any_source, 0, input_data_);
  } else {
    world_.recv(boost::mpi::any_source, 0, input_data_);
    if (input_data_.path[0] == -1) return true;
    input_data_.path.push_back(my_rank);
    if (my_rank != input_data_.target) {
      auto it = std::ranges::find(route, my_rank);
      if (it != route.end() && std::ranges::next(it) != route.end()) {
        world_.send(*std::ranges::next(it), 0, input_data_);
      }
    } else {
      world_.send(0, 0, input_data_);
    }
  }
  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::PostProcessingImpl() {
  world_.barrier();
  if (world_.rank() == 0) {
    *reinterpret_cast<InputData *>(task_data->outputs[0]) = input_data_;
  }
  return true;
}

namespace komshina_d_grid_torus_mpi {

std::vector<int> TestTaskMPI::CalculateRoute(int dest, int sizeX, int sizeY) {
  std::vector<int> route;
  int current = 0;
  int dest_x = dest % sizeX;
  int dest_y = dest / sizeX;
  int cur_x = current % sizeX;
  int cur_y = current / sizeX;

  while (cur_x != dest_x) {
    cur_x = (cur_x + 1) % sizeX;
    route.push_back((cur_y * sizeX) + cur_x);
  }
  while (cur_y != dest_y) {
    cur_y = (cur_y + 1) % sizeY;
    route.push_back((cur_y)*sizeX + cur_x);
  }
  return route;
}
}  // namespace komshina_d_grid_torus_mpi