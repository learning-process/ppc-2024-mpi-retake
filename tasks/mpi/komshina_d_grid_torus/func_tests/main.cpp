#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_mpi, run_pipeline) {
  boost::mpi::communicator world;

  int gridDim = static_cast<int>(std::sqrt(world.size()));
  if (gridDim * gridDim == world.size() && world.size() >= 4) {
    komshina_d_grid_torus_mpi::TestTaskMPI::InputData input("hello, world!", world.size() - 1);
    std::vector<int> expectedPath;
    auto route = komshina_d_grid_torus_mpi::TestTaskMPI::calculate_route(input.target, gridDim, gridDim);
    expectedPath.push_back(0);
    expectedPath.insert(expectedPath.end(), route.begin(), route.end());

    komshina_d_grid_torus_mpi::TestTaskMPI::InputData output;

    auto taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
      taskData->inputs_count.emplace_back(1);
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
      taskData->outputs_count.emplace_back(1);
    }

    auto testTask = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(taskData);
    ASSERT_TRUE(testTask->ValidationImpl());
    testTask->PreProcessingImpl();
    testTask->RunImpl();
    testTask->PostProcessingImpl();

    if (world.rank() == 0) {
      ASSERT_EQ(output.payload, input.payload);
      ASSERT_EQ(output.path, expectedPath);
    }
  }
}

TEST(komshina_d_grid_torus_mpi, invalid_grid_size) {
  boost::mpi::communicator world;

  if (world.size() < 4 ||
      static_cast<int>(std::sqrt(world.size())) * static_cast<int>(std::sqrt(world.size())) != world.size()) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    auto testTask = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(taskData);
    ASSERT_FALSE(testTask->ValidationImpl());
  }
}

TEST(komshina_d_grid_torus_mpi, invalid_target) {
  boost::mpi::communicator world;

  int gridDim = static_cast<int>(std::sqrt(world.size()));
  if (gridDim * gridDim == world.size() && world.size() >= 4) {
    komshina_d_grid_torus_mpi::TestTaskMPI::InputData input("Invalid target", world.size());

    auto taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
      taskData->inputs_count.emplace_back(1);
    }

    auto testTask = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(taskData);
    ASSERT_FALSE(testTask->ValidationImpl());
  }
}

TEST(komshina_d_grid_torus_mpi, message_reaches_target) {
  boost::mpi::communicator world;
  int gridDim = static_cast<int>(std::sqrt(world.size()));
  if (gridDim * gridDim == world.size() && world.size() >= 4) {
    komshina_d_grid_torus_mpi::TestTaskMPI::InputData input("Test Message", world.size() - 1);
    komshina_d_grid_torus_mpi::TestTaskMPI::InputData output;

    auto taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
      taskData->inputs_count.emplace_back(1);
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
      taskData->outputs_count.emplace_back(1);
    }

    auto testTask = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(taskData);
    testTask->RunImpl();

    if (world.rank() == world.size() - 1) {
      ASSERT_EQ(output.payload, input.payload);
    }
  }
}

TEST(komshina_d_grid_torus_mpi, empty_message) {
  boost::mpi::communicator world;
  int gridDim = static_cast<int>(std::sqrt(world.size()));
  if (gridDim * gridDim == world.size() && world.size() >= 4) {
    komshina_d_grid_torus_mpi::TestTaskMPI::InputData input("", world.size() - 1);
    komshina_d_grid_torus_mpi::TestTaskMPI::InputData output;

    auto taskData = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
      taskData->inputs_count.emplace_back(1);
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
      taskData->outputs_count.emplace_back(1);
    }

    auto testTask = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(taskData);
    testTask->RunImpl();

    if (world.rank() == world.size() - 1) {
      ASSERT_EQ(output.payload, input.payload);
    }
  }
}