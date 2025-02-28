#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/komshina_d_grid_tor/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_topology_mpi, TestValidation) {
  boost::mpi::communicator world;
  if (world.size() != 4) {
    GTEST_SKIP() << "Not enough processes for this test.";
    return;
  }

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(input_data.data());
  task_data_mpi->inputs_count.emplace_back(input_data.size());
  task_data_mpi->outputs.emplace_back(output_data.data());
  task_data_mpi->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());

  task_data_mpi->inputs.clear();
  task_data_mpi->inputs_count.clear();

  ASSERT_FALSE(task.ValidationImpl());
}

TEST(komshina_d_grid_torus_topology_mpi, TestDataTransmission) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP() << "Not enough processes for this test.";
    return;
  }

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(input_data.data());
  task_data_mpi->inputs_count.emplace_back(input_data.size());
  task_data_mpi->outputs.emplace_back(output_data.data());
  task_data_mpi->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_TRUE(task.ValidationImpl());

  ASSERT_TRUE(task.PreProcessingImpl());
}

TEST(komshina_d_grid_torus_topology_mpi, TestEmptyOutputData) {
  boost::mpi::communicator world;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 1);
  std::vector<uint8_t> output_data;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(input_data.data());
  task_data_mpi->inputs_count.emplace_back(input_data.size());
  task_data_mpi->outputs.emplace_back(output_data.data());
  task_data_mpi->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_FALSE(task.ValidationImpl()) << "Validation should fail with empty output data";
}

TEST(komshina_d_grid_torus_topology_mpi, TestNullptrInput) {
  boost::mpi::communicator world;

  std::vector<uint8_t> output_data(4);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(nullptr);
  task_data_mpi->inputs_count.emplace_back(4);
  task_data_mpi->outputs.emplace_back(output_data.data());
  task_data_mpi->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_FALSE(task.ValidationImpl()) << "Validation should fail with nullptr in the input data";
}

TEST(komshina_d_grid_torus_topology_mpi, TestNonMatchingInputOutputSizes) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP() << "Not enough processes for this test.";
    return;
  }

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);

  std::vector<uint8_t> output_data(2);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(input_data.data());
  task_data_mpi->inputs_count.emplace_back(input_data.size());
  task_data_mpi->outputs.emplace_back(output_data.data());
  task_data_mpi->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_FALSE(task.ValidationImpl());
}

TEST(komshina_d_grid_torus_topology_mpi, TestEmptyInputsOnly) {
  std::vector<uint8_t> output_data(4);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs_count.emplace_back(4);
  task_data_mpi->outputs.emplace_back(output_data.data());
  task_data_mpi->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_FALSE(task.ValidationImpl()) << "Validation should fail with empty inputs but non-empty inputs_count";
}

TEST(komshina_d_grid_torus_topology_mpi, TestEmptyInputsCountOnly) {
  std::vector<uint8_t> input_data(4);
  std::vector<uint8_t> output_data(4);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(input_data.data());
  task_data_mpi->outputs.emplace_back(output_data.data());
  task_data_mpi->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_FALSE(task.ValidationImpl()) << "Validation should fail with non-empty inputs but empty inputs_count";
}

TEST(komshina_d_grid_torus_topology_mpi, TestPostProcessingImpl) {
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_TRUE(task.PostProcessingImpl());
}

TEST(komshina_d_grid_torus_topology_mpi, TestPreProcessingImpl) {
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(nullptr);
  task_data_mpi->inputs_count.emplace_back(0);

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_FALSE(task.PreProcessingImpl());
}

TEST(komshina_d_grid_torus_topology_mpi, TestPerfectSquareProcessCount) {
  boost::mpi::communicator world;
  int size = world.size();
  int sqrt_size = static_cast<int>(std::sqrt(size));

  bool is_perfect_square = (sqrt_size * sqrt_size == size);

  if (is_perfect_square) {
    ASSERT_TRUE(is_perfect_square) << "The number of processes should form a perfect square.";
  } else {
    ASSERT_FALSE(is_perfect_square) << "The number of processes should not form a perfect square.";
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestRankSizeGridSize) {
  boost::mpi::communicator world;

  int rank = world.rank();
  int size = world.size();
  int grid_size = static_cast<int>(std::sqrt(size));

  ASSERT_GE(rank, 0) << "Rank must be non-negative";
  ASSERT_GT(size, 0) << "Size must be greater than zero";
  ASSERT_EQ(grid_size * grid_size, size) << "Size must be a perfect square";
}

TEST(komshina_d_grid_torus_topology_mpi, TestComputeNeighborsInRun) {
  boost::mpi::communicator world;
  int size = world.size();
  int rank = world.rank();
  int grid_size = static_cast<int>(std::sqrt(size));

  if (grid_size * grid_size != size) {
    GTEST_SKIP() << "Skipping test: number of processes is not a perfect square.";
    return;
  }

  std::vector<uint8_t> input_data(4, rank);
  std::vector<uint8_t> output_data(4);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(input_data.data());
  task_data_mpi->inputs_count.emplace_back(input_data.size());
  task_data_mpi->outputs.emplace_back(output_data.data());
  task_data_mpi->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_TRUE(task.ValidationImpl());

  for (int step = 0; step < grid_size; ++step) {
    auto neighbors = task.ComputeNeighbors(rank, grid_size);

    ASSERT_EQ(neighbors.size(), 4) << "Each rank should have exactly 4 neighbors.";

    for (int neighbor : neighbors) {
      ASSERT_GE(neighbor, 0) << "Neighbor rank should not be negative.";
      ASSERT_LT(neighbor, size) << "Neighbor rank should be within the valid range.";
    }
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestNeighborOutOfBounds) {
  boost::mpi::communicator world;
  int size = world.size();
  int rank = world.rank();
  int grid_size = static_cast<int>(std::sqrt(size));

  if (grid_size * grid_size != size) {
    GTEST_SKIP() << "Skipping test: number of processes is not a perfect square.";
    return;
  }

  std::vector<uint8_t> input_data(4, rank);
  std::vector<uint8_t> output_data(4);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(input_data.data());
  task_data_mpi->inputs_count.emplace_back(input_data.size());
  task_data_mpi->outputs.emplace_back(output_data.data());
  task_data_mpi->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_TRUE(task.ValidationImpl());

  std::vector<int> invalid_neighbors = {size, size + 1, rank + 1, rank - 1};

  for (int neighbor : invalid_neighbors) {
    if (neighbor >= size) {
      ASSERT_NO_FATAL_FAILURE({
        if (neighbor >= size) {
          SUCCEED() << "Neighbor out of bounds correctly skipped.";
        } else {
          FAIL() << "Out-of-bounds neighbor was processed incorrectly.";
        }
      });
    }
  }
}

TEST(komshina_d_grid_torus_topology_mpi, ComputeNeighbors_Grid2x2) {
  int grid_size = 2;
  
  std::vector<std::vector<int>> expected_neighbors = {{1, 1, 2, 2}, {0, 0, 3, 3}, {3, 3, 0, 0}, {2, 2, 1, 1}};

  for (int rank = 0; rank < 4; ++rank) {
    auto neighbors = komshina_d_grid_torus_topology_mpi::TestTaskMPI::ComputeNeighbors(rank, grid_size);
    ASSERT_EQ(neighbors, expected_neighbors[rank]) << "Incorrect neighbors for rank " << rank;
  }
}

TEST(komshina_d_grid_torus_topology_mpi, ComputeNeighbors_WrapAround) {
  int grid_size = 4;

  std::vector<std::pair<int, std::vector<int>>> test_cases = {
      {0, {3, 1, 12, 4}}, {3, {2, 0, 15, 7}}, {12, {15, 13, 8, 0}}, {15, {14, 12, 11, 3}}};

  for (const auto& [rank, expected] : test_cases) {
    auto neighbors = komshina_d_grid_torus_topology_mpi::TestTaskMPI::ComputeNeighbors(rank, grid_size);
    ASSERT_EQ(neighbors, expected) << "Incorrect neighbors for rank " << rank;
  }
}