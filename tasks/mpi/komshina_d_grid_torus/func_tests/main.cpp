#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"


TEST(komshina_d_grid_torus_mpi, Test_Transfer) {
  boost::mpi::communicator world;

  int sqrtN = static_cast<int>(std::sqrt(world.size()));
  if (sqrtN * sqrtN == world.size() && world.size() >= 2) {
    std::vector<int> input{1024, 1};
    std::vector<int> expectedPath{0, 1};

    std::vector<int> output(1, 0);
    std::vector<int> outputPath(world.size(), -1);

    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
      task_data_mpi->inputs_count.emplace_back(input.size());
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPath.data()));
      task_data_mpi->outputs_count.emplace_back(output.size());
      task_data_mpi->outputs_count.emplace_back(outputPath.size());
    }

    komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
    ASSERT_EQ(test_task_mpi.Validation(), true);
    test_task_mpi.PreProcessing();
    test_task_mpi.Run();
    test_task_mpi.PostProcessing();

    if (world.rank() == 0) {
      outputPath.erase(std::remove(outputPath.begin(), outputPath.end(), -1), outputPath.end());
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}

TEST(komshina_d_grid_torus_mpi, Test_Transfer_Neighbor) {
  boost::mpi::communicator world;

  int sqrtN = static_cast<int>(std::sqrt(world.size()));
  if (sqrtN * sqrtN == world.size() && world.size() >= 4) {
    std::vector<int> input{512, 1};
    std::vector<int> expectedPath{0, 1};

    std::vector<int> output(1, 0);
    std::vector<int> outputPath(world.size(), -1);

    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
      task_data_mpi->inputs_count.emplace_back(input.size());
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPath.data()));
      task_data_mpi->outputs_count.emplace_back(output.size());
      task_data_mpi->outputs_count.emplace_back(outputPath.size());
    }

    komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
    ASSERT_EQ(test_task_mpi.Validation(), true);
    test_task_mpi.PreProcessing();
    test_task_mpi.Run();
    test_task_mpi.PostProcessing();

    if (world.rank() == 0) {
      outputPath.erase(std::remove(outputPath.begin(), outputPath.end(), -1), outputPath.end());
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}

TEST(komshina_d_grid_torus_mpi, Test_Transfer_Distant) {
  boost::mpi::communicator world;

  int sqrtN = static_cast<int>(std::sqrt(world.size()));
  if (sqrtN * sqrtN == world.size() && world.size() >= 4) {
    int target_rank = world.size() - 1;
    std::vector<int> input{256, target_rank};
    std::vector<int> expectedPath{0, 1, 2, 3};

    std::vector<int> output(1, 0);
    std::vector<int> outputPath(world.size(), -1);

    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
      task_data_mpi->inputs_count.emplace_back(input.size());
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPath.data()));
      task_data_mpi->outputs_count.emplace_back(output.size());
      task_data_mpi->outputs_count.emplace_back(outputPath.size());
    }

    komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
    ASSERT_EQ(test_task_mpi.Validation(), true);
    test_task_mpi.PreProcessing();
    test_task_mpi.Run();
    test_task_mpi.PostProcessing();

    if (world.rank() == 0) {
      outputPath.erase(std::remove(outputPath.begin(), outputPath.end(), -1), outputPath.end());
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}

TEST(komshina_d_grid_torus_mpi, Test_Transfer_Self) {
  boost::mpi::communicator world;

  int sqrtN = static_cast<int>(std::sqrt(world.size()));
  if (sqrtN * sqrtN == world.size() && world.size() >= 2) {
    std::vector<int> input{128, 0};
    std::vector<int> expectedPath{0};

    std::vector<int> output(1, 0);
    std::vector<int> outputPath(world.size(), -1);

    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
      task_data_mpi->inputs_count.emplace_back(input.size());
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPath.data()));
      task_data_mpi->outputs_count.emplace_back(output.size());
      task_data_mpi->outputs_count.emplace_back(outputPath.size());
    }

    komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
    ASSERT_EQ(test_task_mpi.Validation(), true);
    test_task_mpi.PreProcessing();
    test_task_mpi.Run();
    test_task_mpi.PostProcessing();

    if (world.rank() == 0) {
      outputPath.erase(std::remove(outputPath.begin(), outputPath.end(), -1), outputPath.end());
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}
TEST(komshina_d_grid_torus_mpi, Test_Validation_Fail) {
  boost::mpi::communicator world;

  int sqrtN = static_cast<int>(std::sqrt(world.size()));
  if (sqrtN * sqrtN != world.size()) {
    std::vector<int> input{64, 1};
    std::vector<int> output(1, 0);
    std::vector<int> outputPath(world.size(), -1);

    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
      task_data_mpi->inputs_count.emplace_back(input.size());
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPath.data()));
      task_data_mpi->outputs_count.emplace_back(output.size());
      task_data_mpi->outputs_count.emplace_back(outputPath.size());
    }

    komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
    ASSERT_EQ(test_task_mpi.Validation(), false);
  }
}

TEST(komshina_d_grid_torus_mpi, Test_Transfer_Distant_Closed_Edges) {
  boost::mpi::communicator world;

  int sqrtN = static_cast<int>(std::sqrt(world.size()));
  if (sqrtN * sqrtN == world.size() && world.size() >= 4) {
    int target_rank = world.size() - 1;
    std::vector<int> input{256, target_rank};
    std::vector<int> expectedPath{0, 1, 2, 3};

    std::vector<int> output(1, 0);
    std::vector<int> outputPath(world.size(), -1);

    std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
      task_data_mpi->inputs_count.emplace_back(input.size());
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPath.data()));
      task_data_mpi->outputs_count.emplace_back(output.size());
      task_data_mpi->outputs_count.emplace_back(outputPath.size());
    }

    komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
    ASSERT_EQ(test_task_mpi.Validation(), true);
    test_task_mpi.PreProcessing();
    test_task_mpi.Run();
    test_task_mpi.PostProcessing();

    if (world.rank() == 0) {
      outputPath.erase(std::remove(outputPath.begin(), outputPath.end(), -1), outputPath.end());
      ASSERT_EQ(output[0], input[0]);
      ASSERT_EQ(outputPath, expectedPath);
    }
  }
}