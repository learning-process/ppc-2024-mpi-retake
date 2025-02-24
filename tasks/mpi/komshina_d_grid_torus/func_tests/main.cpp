#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_mpi, Test_Tran) {
  boost::mpi::communicator world;
  int world_size = world.size();
  int grid_width = std::sqrt(world_size);

  if (grid_width * grid_width == world_size && world_size >= 4) {
    std::string message = "Hello, MPI!";
    int destination = world_size - 1;

    std::vector<int> expectedPath = komshina_d_grid_torus_mpi::TestTaskMPI::calculate_route(0, destination, grid_width);

    komshina_d_grid_torus_mpi::TestTaskMPI::InputData input_data(message, destination);
    komshina_d_grid_torus_mpi::TestTaskMPI::InputData output_data;

    std::vector<uint8_t> input(reinterpret_cast<uint8_t*>(&input_data),
                               reinterpret_cast<uint8_t*>(&input_data) + sizeof(input_data));
    std::vector<uint8_t> output(sizeof(output_data));

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(input.data());
      taskDataPar->inputs_count.emplace_back(input.size());
      taskDataPar->outputs.emplace_back(output.data());
      taskDataPar->outputs_count.emplace_back(output.size());
    }

    komshina_d_grid_torus_mpi::TestTaskMPI GridTorusMPI(taskDataPar);
    ASSERT_TRUE(GridTorusMPI.ValidationImpl());
    GridTorusMPI.PreProcessingImpl();
    GridTorusMPI.RunImpl();
    GridTorusMPI.PostProcessingImpl();

    if (world.rank() == 0) {
      output_data = *reinterpret_cast<komshina_d_grid_torus_mpi::TestTaskMPI::InputData*>(output.data());
      ASSERT_EQ(output_data.payload, std::vector<char>(message.begin(), message.end()));
      ASSERT_EQ(output_data.path, expectedPath);
    }
  }
}

TEST(komshina_d_grid_torus_mpi, small_grid_processes) {
  boost::mpi::communicator world;
  if (world.size() != 4) {
    GTEST_SKIP();
    return;
  }

  std::string data = "abcd";
  int dest = 1;
  std::vector<int> route_expected = {0, 1};

  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_in_struct(data, dest);
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_out_struct;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data_in_struct));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&data_out_struct));
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(task_data_mpi);
  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(data_out_struct.path, route_expected);
    ASSERT_EQ(std::string(data_out_struct.payload.begin(), data_out_struct.payload.end()), data);
  }
}

TEST(komshina_d_grid_torus_mpi_test, test_preprocessing_invalid_target) {
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData input_data;
  input_data.target = 20;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_data));

  std::vector<uint8_t> output_data;
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_mpi->outputs_count.emplace_back(output_data.size());

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_TRUE(test_task_mpi.Validation());

  EXPECT_FALSE(test_task_mpi.PreProcessing());
}

TEST(komshina_d_grid_torus_mpi, test_same_start_and_dest) {
  int width = 4;
  int start = 2;

  std::vector<int> path = komshina_d_grid_torus_mpi::TestTaskMPI::calculate_route(start, start, width);
  EXPECT_EQ(path.size(), 1);
  EXPECT_EQ(path.front(), start);
}

TEST(komshina_d_grid_torus_mpi, test_task_mpi_invalid_target) {
  std::string input_str = "Test";
  int target_node = 100;
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData in_data(input_str, target_node);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in_data));
  task_data_mpi->inputs_count.emplace_back(1);

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  EXPECT_FALSE(test_task_mpi.Validation());
}

TEST(komshina_d_grid_torus_mpi, test_task_mpi_no_input) {
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  EXPECT_FALSE(test_task_mpi.Validation());
}

TEST(komshina_d_grid_torus_mpi, preprocessing_check) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  std::string data = "aaabbbcccddd";
  int dest = 1;
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_in_struct(data, dest);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&data_in_struct));
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_TRUE(task.PreProcessingImpl());
}

TEST(komshina_d_grid_torus_mpi, run_impl_check) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  std::string data = "aaabbbcccddd";
  int dest = 1;
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_in_struct(data, dest);
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_out_struct;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&data_in_struct));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&data_out_struct));
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(task_data_mpi);

  task.RunImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(std::string(data_out_struct.payload.begin(), data_out_struct.payload.end()), data);
  }
}

TEST(komshina_d_grid_torus_mpi, postprocessing_check) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  std::string data = "aaabbbcccddd";
  int dest = 1;
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_in_struct(data, dest);
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_out_struct;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&data_in_struct));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&data_out_struct));
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(task_data_mpi);
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(std::string(data_out_struct.payload.begin(), data_out_struct.payload.end()), data);
  }
}