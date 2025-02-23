#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_mpi, run_pipeline) {
  boost::mpi::communicator world;

  int gridDim = static_cast<int>(std::sqrt(world.size()));
  if (gridDim * gridDim == world.size() && world.size() >= 4) {
    komshina_d_grid_torus_mpi::TestTaskMPI::InputData input("hello, world!", world.size() - 1);
    std::vector<int> expected_path;
    auto route = komshina_d_grid_torus_mpi::TestTaskMPI::CalculateRoute(input.target, gridDim, gridDim);
    expected_path.push_back(0);
    expected_path.insert(expected_path.end(), route.begin(), route.end());

    komshina_d_grid_torus_mpi::TestTaskMPI::InputData output;

    auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
      task_data_mpi->inputs_count.emplace_back(1);
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
      task_data_mpi->outputs_count.emplace_back(1);
    }

    auto test_task_mpi = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(task_data_mpi);
    ASSERT_TRUE(test_task_mpi->ValidationImpl());
    test_task_mpi->PreProcessingImpl();
    test_task_mpi->RunImpl();
    test_task_mpi->PostProcessingImpl();

    if (world.rank() == 0) {
      ASSERT_EQ(output.payload, input.payload);
      ASSERT_EQ(output.path, expected_path);
    }
  }
}

TEST(komshina_d_grid_torus_mpi, invalid_grid_size) {
  boost::mpi::communicator world;

  if (world.size() < 4 ||
      static_cast<int>(std::sqrt(world.size())) * static_cast<int>(std::sqrt(world.size())) != world.size()) {
    auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
    auto test_task_mpi = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(task_data_mpi);
    ASSERT_FALSE(test_task_mpi->ValidationImpl());
  }
}

TEST(komshina_d_grid_torus_mpi, invalid_target) {
  boost::mpi::communicator world;

  int gridDim = static_cast<int>(std::sqrt(world.size()));
  if (gridDim * gridDim == world.size() && world.size() >= 4) {
    komshina_d_grid_torus_mpi::TestTaskMPI::InputData input("Invalid target", world.size());

    auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
      task_data_mpi->inputs_count.emplace_back(1);
    }

    auto test_task_mpi = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(task_data_mpi);
    ASSERT_FALSE(test_task_mpi->ValidationImpl());
  }
}

TEST(komshina_d_grid_torus_mpi, message_reaches_target) {
  boost::mpi::communicator world;
  int gridDim = static_cast<int>(std::sqrt(world.size()));
  if (gridDim * gridDim == world.size() && world.size() >= 4) {
    komshina_d_grid_torus_mpi::TestTaskMPI::InputData input("Test Message", world.size() - 1);
    komshina_d_grid_torus_mpi::TestTaskMPI::InputData output;

    auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
      task_data_mpi->inputs_count.emplace_back(1);
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
      task_data_mpi->outputs_count.emplace_back(1);
    }

    auto test_task_mpi = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(task_data_mpi);
    test_task_mpi->RunImpl();

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

    auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
      task_data_mpi->inputs_count.emplace_back(1);
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
      task_data_mpi->outputs_count.emplace_back(1);
    }

    auto test_task_mpi = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(task_data_mpi);
    test_task_mpi->RunImpl();

    if (world.rank() == world.size() - 1) {
      ASSERT_EQ(output.payload, input.payload);
    }
  }
}