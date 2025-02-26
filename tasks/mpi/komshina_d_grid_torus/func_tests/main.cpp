#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_mpi, validation_failed_wrong_task_data) {
  boost::mpi::communicator world;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_FALSE(test_task_mpi.ValidationImpl());
  }
}

TEST(komshina_d_grid_torus_mpi, check_out_of_bounds_destination) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData in("out of bounds", -1);
  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData out;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&in));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_FALSE(test_task_mpi.PreProcessingImpl());
}

TEST(komshina_d_grid_torus_mpi, route_calculation_same_source_and_destination) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  int source_rank = 0;
  int dest_rank = 0;

  std::vector<int> expected_route = {source_rank};

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  std::vector<int> route = komshina_d_grid_torus_mpi::TestTaskMPI::CalculateRoute(dest_rank, 3, 3);
  ASSERT_EQ(route, expected_route);
}

TEST(komshina_d_grid_torus_mpi, SimpleDataTransmission) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  std::vector<int> inputData{1337, 1};
  std::vector<int> expectedRoute{0, 1};
  std::vector<int> outputData(1, 0);
  std::vector<int> actualRoute;

  size_t routeSize = std::sqrt(world.size());
  actualRoute.reserve(routeSize);
  for (size_t i = 0; i < routeSize; ++i) {
    actualRoute.push_back(-1);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    task_data_mpi->inputs_count.emplace_back(inputData.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
    task_data_mpi->outputs_count.emplace_back(outputData.size());
    task_data_mpi->outputs_count.emplace_back(actualRoute.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    actualRoute = test_task_mpi.CalculateRoute(inputData[1], std::sqrt(world.size()), std::sqrt(world.size()));
    ASSERT_EQ(outputData[0], inputData[0]);
    ASSERT_EQ(actualRoute, expectedRoute);
  }
}

TEST(komshina_d_grid_torus_mpi, ComplexRouteCheck) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  std::vector<int> inputData{1337, 3};
  std::vector<int> expectedRoute{0, 1, 3};
  std::vector<int> outputData(1, 0);
  std::vector<int> actualRoute;

  size_t routeSize = std::sqrt(world.size());
  actualRoute.reserve(routeSize);
  for (size_t i = 0; i < routeSize; ++i) {
    actualRoute.push_back(-1);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    task_data_mpi->inputs_count.emplace_back(inputData.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
    task_data_mpi->outputs_count.emplace_back(outputData.size());
    task_data_mpi->outputs_count.emplace_back(actualRoute.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    actualRoute = test_task_mpi.CalculateRoute(inputData[1], std::sqrt(world.size()), std::sqrt(world.size()));
    ASSERT_EQ(outputData[0], inputData[0]);
    ASSERT_EQ(actualRoute, expectedRoute);
  }
}

TEST(komshina_d_grid_torus_mpi, InvalidTargetRankNegative) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  std::vector<int> inputData{1337, -5};
  std::vector<int> outputData(1, 0);
  std::vector<int> actualRoute;

  size_t routeSize = std::sqrt(world.size());
  actualRoute.reserve(routeSize);
  for (size_t i = 0; i < routeSize; ++i) {
    actualRoute.push_back(-1);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    task_data_mpi->inputs_count.emplace_back(inputData.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
    task_data_mpi->outputs_count.emplace_back(outputData.size());
    task_data_mpi->outputs_count.emplace_back(actualRoute.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI TopologyTorusMPI(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(TopologyTorusMPI.ValidationImpl(), false);
  }
}

TEST(komshina_d_grid_torus_mpi, SelfMessagePassing) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  std::vector<int> inputData{1337, 0};
  std::vector<int> expectedRoute{0};
  std::vector<int> outputData(1, 0);
  std::vector<int> actualRoute;

  size_t routeSize = std::sqrt(world.size());
  actualRoute.reserve(routeSize);
  for (size_t i = 0; i < routeSize; ++i) {
    actualRoute.push_back(-1);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    task_data_mpi->inputs_count.emplace_back(inputData.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(actualRoute.data()));
    task_data_mpi->outputs_count.emplace_back(outputData.size());
    task_data_mpi->outputs_count.emplace_back(actualRoute.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    actualRoute = test_task_mpi.CalculateRoute(inputData[1], std::sqrt(world.size()), std::sqrt(world.size()));
    ASSERT_EQ(outputData[0], inputData[0]);
    ASSERT_EQ(actualRoute, expectedRoute);
  }
}