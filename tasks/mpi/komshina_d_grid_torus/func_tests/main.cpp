#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef> 
#include <cstdint>
#include <memory>
#include <numeric>
#include <cstring>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

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
    ASSERT_FALSE(task.ValidationImpl()) << "Validation should fail with insufficient input data";
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

TEST(komshina_d_grid_torus_topology_mpi, TestNonSquareTopology) {
  boost::mpi::communicator world;

  int size = world.size();
  int sqrt_size = static_cast<int>(std::sqrt(size));

  if (sqrt_size * sqrt_size != size) {
    std::vector<uint8_t> input_data(4);
    std::vector<uint8_t> output_data(4);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(input_data.data());
    task_data->inputs_count.emplace_back(input_data.size());
    task_data->outputs.emplace_back(output_data.data());
    task_data->outputs_count.emplace_back(output_data.size());

    komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data);

    ASSERT_FALSE(task.ValidationImpl()) << "Validation should fail for a non-square topology";
  }
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

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], input_data[i]) << "Mismatch at index " << i;
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestEmptyOutputData) {
  boost::mpi::communicator world;

  std::vector<uint8_t> input_data(4);
  std::iota(input_data.begin(), input_data.end(), 1);
  std::vector<uint8_t> output_data;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data);

  ASSERT_FALSE(task.ValidationImpl()) << "Validation should fail with empty output data";
}

TEST(komshina_d_grid_torus_topology_mpi, TestNullptrInput) {
  boost::mpi::communicator world;

  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(nullptr);
  task_data->inputs_count.emplace_back(4);
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data);

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

  ASSERT_TRUE(task.RunImpl());

  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], input_data[i]);
  }
}

TEST(komshina_d_grid_torus_topology_mpi, TestEmptyInputsOnly) {
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(4);
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data);

  ASSERT_FALSE(task.ValidationImpl()) << "Validation should fail with empty inputs but non-empty inputs_count";
}

TEST(komshina_d_grid_torus_topology_mpi, TestEmptyInputsCountOnly) {
  std::vector<uint8_t> input_data(4);
  std::vector<uint8_t> output_data(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input_data.data());
  task_data->outputs.emplace_back(output_data.data());
  task_data->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_topology_mpi::TestTaskMPI task(task_data);

  ASSERT_FALSE(task.ValidationImpl()) << "Validation should fail with non-empty inputs but empty inputs_count";
}

TEST(komshina_d_grid_torus_topology_mpi, TestSmallOutputBuffer) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
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

  ASSERT_TRUE(task.ValidationImpl());

  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());

  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], 0) << "Output buffer should remain unchanged due to insufficient size";
  }
}