#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include <memory>

#include "core/task/include/task.hpp"
#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_mpi, validation_check) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  komshina_d_grid_torus_mpi::TestTaskMPI task(taskDataPar);

  ASSERT_TRUE(task.ValidationImpl());
}

TEST(komshina_d_grid_torus_mpi, preprocessing_check) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;

  std::string data = "aaabbbcccddd";
  int dest = 1;
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_in_struct(data, dest);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data_in_struct));
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(taskDataPar);

  ASSERT_TRUE(task.PreProcessingImpl());
}

TEST(komshina_d_grid_torus_mpi, run_impl_check) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;

  std::string data = "aaabbbcccddd";
  int dest = 1;
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_in_struct(data, dest);
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_out_struct;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data_in_struct));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&data_out_struct));
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(taskDataPar);

  task.RunImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(std::string(data_out_struct.payload.begin(), data_out_struct.payload.end()), data);
  }
}

TEST(komshina_d_grid_torus_mpi, postprocessing_check) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;

  std::string data = "aaabbbcccddd";
  int dest = 1;
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_in_struct(data, dest);
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_out_struct;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data_in_struct));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&data_out_struct));
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(taskDataPar);
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(std::string(data_out_struct.payload.begin(), data_out_struct.payload.end()), data);
  }
}

TEST(komshina_d_grid_torus_mpi, small_grid_4_processes) {
  boost::mpi::communicator world;
  if (world.size() != 4) return;

  std::string data = "abcd";
  int dest = 1;
  std::vector<int> route_expected = {0, 1};

  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_in_struct(data, dest);
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_out_struct;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data_in_struct));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&data_out_struct));
    taskDataPar->outputs_count.emplace_back(1);
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(taskDataPar);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());

  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(data_out_struct.path, route_expected);
    ASSERT_EQ(std::string(data_out_struct.payload.begin(), data_out_struct.payload.end()), data);
  }
}

TEST(komshina_d_grid_torus_mpi, non_square_grid) {
  boost::mpi::communicator world;
  if (world.size() != 5) return;

  std::string data = "xyzw";
  int dest = 4;
  std::vector<int> route_expected = {0, 1, 2, 3, 4};

  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_in_struct(data, dest);
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_out_struct;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data_in_struct));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&data_out_struct));
    taskDataPar->outputs_count.emplace_back(1);
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(taskDataPar);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());

  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(data_out_struct.path, route_expected);
    ASSERT_EQ(std::string(data_out_struct.payload.begin(), data_out_struct.payload.end()), data);
  }
}

TEST(komshina_d_grid_torus_mpi, invalid_target_rank) {
  boost::mpi::communicator world;
  if (world.size() < 4) return;

  std::string data = "testdata";
  int dest = -1;
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_in_struct(data, dest);
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_out_struct;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data_in_struct));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&data_out_struct));
    taskDataPar->outputs_count.emplace_back(1);
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(taskDataPar);

  ASSERT_FALSE(task.ValidationImpl());
}

TEST(komshina_d_grid_torus_mpi, large_grid_16_processes) {
  boost::mpi::communicator world;
  if (world.size() != 16) return;

  std::string data = "abcdefghijklmno";
  int dest = 15;
  std::vector<int> route_expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_in_struct(data, dest);
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData data_out_struct;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data_in_struct));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&data_out_struct));
    taskDataPar->outputs_count.emplace_back(1);
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(taskDataPar);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());

  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(data_out_struct.path, route_expected);
    ASSERT_EQ(std::string(data_out_struct.payload.begin(), data_out_struct.payload.end()), data);
  }
}