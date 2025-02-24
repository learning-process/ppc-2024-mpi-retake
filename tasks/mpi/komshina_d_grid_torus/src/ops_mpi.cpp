#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

bool komshina_d_grid_torus_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto *in_ptr = reinterpret_cast<InputData *>(task_data->inputs[0]);
    input_data_ = *in_ptr;
    if (input_data_.target >= world_.size() || input_data_.target < 0) {
      return false;
    }
    input_data_.path.clear();
  }

  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::ValidationImpl() {
  int world_size = world_.size();
  int grid_width = std::sqrt(world_size);

  if (grid_width * grid_width != world_size) {
    return false;
  }
  if (world_.rank() == 0) {
    if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1 || task_data->outputs_count.size() != 1) {
      return false;
    }
  }

  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::RunImpl() {
  int my_rank = world_.rank();

  if (my_rank == 0) {
    input_data_.path = calculate_route(0, input_data_.target, std::sqrt(world_.size()));

    int next = input_data_.path[1];
    world_.send(next, 0, input_data_);
    world_.recv(boost::mpi::any_source, 0, input_data_);
  } else {
    world_.recv(boost::mpi::any_source, 0, input_data_);
    input_data_.path.push_back(my_rank);

    if (my_rank != input_data_.target) {
      int next = input_data_.path[input_data_.path.size() - 1];
      world_.send(next, 0, input_data_);
    } else {
      world_.send(0, 0, input_data_);
    }
  }

  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::PostProcessingImpl() {
  world_.barrier();
  if (world_.rank() == 0) {
    auto *dataOutPtr = reinterpret_cast<InputData *>(task_data->outputs[0]);
    *dataOutPtr = input_data_;
  }
  return true;
}

namespace komshina_d_grid_torus_mpi {
std::vector<int> TestTaskMPI::calculate_route(int start, int dest, int width) {
  std::vector<int> path = {start};
  int current = start;

  while (current != dest) {
    int x = current % width;
    int y = current / width;
    int dest_x = dest % width;
    int dest_y = dest / width;

    if (x != dest_x) {
      x = (x + 1) % width;
      if (x == 0) {
        x = width - 1;
      }
    }
    if (y != dest_y) {
      y = (y + 1) % width;
      if (y == 0) {
        y = width - 1;
      }
    }

    current = x + y * width;
    path.push_back(current);
  }

  return path;
}

}  // namespace komshina_d_grid_torus_mpi