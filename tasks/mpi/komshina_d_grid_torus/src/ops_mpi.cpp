#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <vector>

bool komshina_d_grid_torus_mpi::TestTaskMPI::PreProcessingImpl() {
  rankX = world_.rank() % sizeX;
  rankY = world_.rank() / sizeX;

  left = (rankX - 1 + sizeX) % sizeX + rankY * sizeX;
  right = (rankX + 1) % sizeX + rankY * sizeX;
  up = rankX + ((rankY - 1 + sizeY) % sizeY) * sizeX;
  down = rankX + ((rankY + 1) % sizeY) * sizeX;

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
  sizeX = std::sqrt(world_size);
  sizeY = world_size / sizeX;
  if (sizeX * sizeY != world_size) {
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
  int destX = task_data_.target % sizeX;
  int destY = task_data_.target / sizeX;

  auto DetermineNext = [this, rank, destX, destY]() {
    if (rankX != destX) {
      return (rankX < destX) ? right : left;
    }
    if (rankY != destY) {
      return (rankY < destY) ? down : up;
    }
    return -1;
  };

  if (rank == 0) {
    task_data_.path.emplace_back(0);
    int NextHop = DetermineNext();
    world_.send(NextHop, 0, task_data_.payload);

    world_.recv(boost::mpi::any_source, 0, task_data_.payload);
  } else {
    std::vector<char> buffer;
    world_.recv(boost::mpi::any_source, 0, buffer);

    task_data_.payload = std::move(buffer);
    task_data_.path.emplace_back(rank);

    int NextHop = (rank == task_data_.target) ? 0 : DetermineNext();
    world_.send(NextHop, 0, task_data_.payload);
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

std::vector<int> komshina_d_grid_torus_mpi::TestTaskMPI::CalculateRoute(int dest, int sizeX, int sizeY) {
  std::vector<int> path = {0};
  int currentX = 0, currentY = 0;
  int destX = dest % sizeX, destY = dest / sizeX;

  while (currentX != destX || currentY != destY) {
    if (currentX != destX) {
      currentX = (currentX + 1) % sizeX;
    } else {
      currentY = (currentY + 1) % sizeY;
    }
    path.push_back(currentX + currentY * sizeX);
  }
  return path;
}
