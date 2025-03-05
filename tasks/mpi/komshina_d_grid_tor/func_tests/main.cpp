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

TEST(komshina_d_grid_torus_topology_mpi, TestInsufficientNodes) {
  boost::mpi::communicator world;

  int size = world.size();
  int sqrt_size = static_cast<int>(std::sqrt(size));
  if (sqrt_size * sqrt_size != size) {
    std::vector<uint8_t> input_data(4);
    std::iota(input_data.begin(), input_data.end(), 9);
    std::vector<uint8_t> output_data(4);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(input_data.data());
    task_data->inputs_count.emplace_back(input_data.size());
    task_data->outputs.emplace_back(output_data.data());
    task_data->outputs_count.emplace_back(output_data.size());

    komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data);
    ASSERT_FALSE(task.ValidationImpl());
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestValidation) {
  boost::mpi::communicator world;
  if (world.size() != 4) {
    GTEST_SKIP() << "Not enough processes for this test.";
    return;
  }

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());

  task_data->inputs.clear();
  task_data->inputs_count.clear();

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

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data);

  ASSERT_TRUE(task.ValidationImpl());

  ASSERT_TRUE(task.PreProcessingImpl());
}

TEST(komshina_d_grid_torus_topology_mpi, TestLargeData) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP() << "Not enough processes for this test.";
    return;
  }

  size_t large_size = 1000;
  std::vector<uint8_t> input_data(large_size);
  std::iota(input_data.begin(), input_data.end(), 0);
  std::vector<uint8_t> output_data(large_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data);

  ASSERT_TRUE(task.ValidationImpl()) << "Validation failed.";
  ASSERT_TRUE(task.PreProcessingImpl()) << "Pre-processing failed.";
  ASSERT_TRUE(task.RunImpl()) << "Run failed.";
  ASSERT_TRUE(task.PostProcessingImpl()) << "Post-processing failed.";

  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], input_data[i]) << "Mismatch at index " << i;
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestNonMatchingInputOutputSizes) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    GTEST_SKIP() << "Not enough processes for this test.";
    return;
  }

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);

  std::vector<uint8_t> output_data(2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data);

  ASSERT_FALSE(task.ValidationImpl());
}

TEST(komshina_d_grid_torus_topology_mpi, TestSmallNumberOfProcesses) {
  boost::mpi::communicator world;
  if (world.size() < 2) {
    GTEST_SKIP() << "Not enough processes for this test.";
    return;
  }

  int sqrt_size = static_cast<int>(std::sqrt(world.size()));
  if (sqrt_size * sqrt_size != world.size()) {
    GTEST_SKIP() << "Test requires a square number of processes.";
    return;
  }

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 9);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());

  ASSERT_TRUE(task.RunImpl());

  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], input_data[i]);
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestNonSquareTopology) {
  boost::mpi::communicator world;

  int size = world.size();
  int sqrt_size = static_cast<int>(std::sqrt(size));

  if (sqrt_size * sqrt_size != size) {
    std::vector<uint8_t> input_data(4);
    std::iota(input_data.begin(), input_data.end(), 9);
    std::vector<uint8_t> output_data(4);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(input_data.data());
    task_data->inputs_count.emplace_back(input_data.size());
    task_data->outputs.emplace_back(output_data.data());
    task_data->outputs_count.emplace_back(output_data.size());

    komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data);

    ASSERT_FALSE(task.ValidationImpl()) << "Validation should fail for non-square topology.";

    ASSERT_FALSE(task.PreProcessingImpl()) << "Pre-processing should fail for non-square topology.";
  }
}