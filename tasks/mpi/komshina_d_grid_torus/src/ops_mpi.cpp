#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

bool komshina_d_grid_torus_mpi::TestTaskMPI::PreProcessingImpl() {
  if (world.rank() == 0) {
    grid_columns = static_cast<int>(std::sqrt(world.size()));
    grid_rows = grid_columns;

    int *input_data = reinterpret_cast<int *>(task_data->inputs[0]);
    routing_packet.message_payload = input_data[0];
    routing_packet.target_rank = input_data[1];
    routing_packet.is_routing_complete = false;
    routing_packet.routing_path.clear();
  }
  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::ValidationImpl() {
  if (world.rank() == 0) {
    int sqrtN = static_cast<int>(std::sqrt(world.size()));

    if (sqrtN * sqrtN == world.size()) {
      return true;
    }

    if (reinterpret_cast<int *>(task_data->inputs[0])[1] < world.size()) {
      return true;
    }

    return false;
  }

  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::RunImpl() {
  if (world.rank() == 0) {
    routing_packet.routing_path = {world.rank()};

    if (routing_packet.target_rank != 0) {
      const int next_hop = GridTorus(0, routing_packet.target_rank, grid_columns, grid_rows, true, true);
      world.send(next_hop, 0, routing_packet);
    } else {
      routing_packet.is_routing_complete = true;
    }
  } else {
    world.recv(boost::mpi::any_source, 0, routing_packet);

    if (!routing_packet.is_routing_complete) {
      routing_packet.routing_path.push_back(world.rank());

      if (world.rank() == routing_packet.target_rank) {
        routing_packet.is_routing_complete = true;
        world.send(0, 0, routing_packet);
      } else {
        const int next_hop = GridTorus(world.rank(), routing_packet.target_rank, grid_columns, grid_rows, true, true);
        world.send(next_hop, 0, routing_packet);
      }
    }
  }
  return true;
}

bool komshina_d_grid_torus_mpi::TestTaskMPI::PostProcessingImpl() {
  world.barrier();

  if (world.rank() == 0) {
    int *output_data = reinterpret_cast<int *>(task_data->outputs[0]);
    output_data[0] = routing_packet.message_payload;

    int *output_path = reinterpret_cast<int *>(task_data->outputs[1]);
    std::copy(routing_packet.routing_path.begin(), routing_packet.routing_path.end(), output_path);
  }
  return true;
}

int komshina_d_grid_torus_mpi::GridTorus(int sourceRank, int targetRank, int gridSizeX, int gridSizeY,
                                         bool isHorizontalClosed, bool isVerticalClosed) {
  int sourceX = sourceRank % gridSizeX;
  int sourceY = sourceRank / gridSizeX;
  int targetX = targetRank % gridSizeX;
  int targetY = targetRank / gridSizeX;

  int dx = targetX - sourceX;
  int dy = targetY - sourceY;

  // Учет замкнутости горизонтальных краев
  if (isHorizontalClosed) {
    if (dx > gridSizeX / 2) dx -= gridSizeX;
    if (dx < -gridSizeX / 2) dx += gridSizeX;
  }

  // Учет замкнутости вертикальных краев
  if (isVerticalClosed) {
    if (dy > gridSizeY / 2) dy -= gridSizeY;
    if (dy < -gridSizeY / 2) dy += gridSizeY;
  }

  // Выбор направления
  if (std::abs(dx) > std::abs(dy)) {
    return (dx > 0) ? sourceRank + 1 : sourceRank - 1;
  } else {
    return (dy > 0) ? sourceRank + gridSizeX : sourceRank - gridSizeX;
  }
}