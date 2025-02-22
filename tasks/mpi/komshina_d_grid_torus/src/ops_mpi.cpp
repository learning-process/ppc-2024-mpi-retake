#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <memory>
#include <vector>

bool komshina_d_grid_torus_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world.rank() == 0) {
    auto *in_ptr = reinterpret_cast<InputData *>(task_data->inputs[0]);
    inputData = *in_ptr;
    inputData.path.clear();
    inputData.path.shrink_to_fit();
  }
  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::ValidationImpl() {
  int worldSize = world.size();

  if (worldSize < 4) return false;

  int gridDim = static_cast<int>(std::sqrt(worldSize));
  if (gridDim * gridDim != worldSize) return false;

  if (world.rank() == 0) {

    if (task_data->inputs.size() != 1 || task_data->outputs_count.size() != 1) return false;

    auto *in_ptr = reinterpret_cast<InputData *>(task_data->inputs[0]);
    if (in_ptr->target >= worldSize || in_ptr->target < 0) {
      return false;
    }
  }

  sizeX = gridDim;
  sizeY = gridDim;

  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::RunImpl() {
  int my_rank = world.rank();
  auto route = calculate_route(inputData.target, sizeX, sizeY);

  if (world.rank() == 0) {
    inputData.path.push_back(0);
    world.send(route[1], 0, inputData);
    world.recv(boost::mpi::any_source, 0, inputData);
  } else {
    world.recv(boost::mpi::any_source, 0, inputData);
    if (inputData.path[0] == -1) return true;
    inputData.path.push_back(my_rank);
    if (my_rank != inputData.target) {
      auto it = std::find(route.begin(), route.end(), my_rank);
      if (it != route.end() && std::next(it) != route.end()) {
        world.send(*std::next(it), 0, inputData);
      }
    } else {
      world.send(0, 0, inputData);
    }
  }
  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::PostProcessingImpl() {
  world.barrier();
  if (world.rank() == 0) {
    *reinterpret_cast<InputData *>(task_data->outputs[0]) = inputData;
  }
  return true;
}

namespace komshina_d_grid_torus_mpi {

std::vector<int> TestTaskMPI::calculate_route(int dest, int sizeX, int sizeY) {
  std::vector<int> route;
  int current = 0;
  int destX = dest % sizeX, destY = dest / sizeX;
  int curX = current % sizeX, curY = current / sizeX;

  while (curX != destX) {
    curX = (curX + 1) % sizeX;
    route.push_back(curY * sizeX + curX);
  }
  while (curY != destY) {
    curY = (curY + 1) % sizeY;
    route.push_back(curY * sizeX + curX);
  }
  return route;
}
}  // namespace komshina_d_grid_torus_mpi