#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <cmath>
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

TEST(komshina_d_grid_torus_mpi, ComplexRouteCheck) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  std::vector<int> input_data{1337, 3};
  std::vector<int> expected_route{0, 1, 3};
  std::vector<int> output_data(1, 0);
  std::vector<int> actual_route;

  size_t route_size = std::sqrt(world.size());
  actual_route.reserve(route_size);
  for (size_t i = 0; i < route_size; ++i) {
    actual_route.push_back(-1);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(input_data.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(actual_route.data()));
    task_data_mpi->outputs_count.emplace_back(output_data.size());
    task_data_mpi->outputs_count.emplace_back(actual_route.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    actual_route = komshina_d_grid_torus_mpi::TestTaskMPI::CalculateRoute(
        input_data[1], std::sqrt(world.size()), std::sqrt(world.size()));
    ASSERT_EQ(output_data[0], input_data[0]);
    ASSERT_EQ(actual_route, expected_route);
  }
}

TEST(komshina_d_grid_torus_mpi, InvalidTargetRankNegative) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  std::vector<int> input_data{1337, -5};
  std::vector<int> output_data(1, 0);
  std::vector<int> actual_route;

  size_t route_size = std::sqrt(world.size());
  actual_route.reserve(route_size);
  for (size_t i = 0; i < route_size; ++i) {
    actual_route.push_back(-1);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(input_data.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(actual_route.data()));
    task_data_mpi->outputs_count.emplace_back(output_data.size());
    task_data_mpi->outputs_count.emplace_back(actual_route.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), false);
  }
}

TEST(komshina_d_grid_torus_mpi, SelfMessagePassing) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  std::vector<int> input_data{1337, 0};
  std::vector<int> expected_route{0};
  std::vector<int> output_data(1, 0);
  std::vector<int> actual_route;

  size_t route_size = std::sqrt(world.size());
  actual_route.reserve(route_size);
  for (size_t i = 0; i < route_size; ++i) {
    actual_route.push_back(-1);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(input_data.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(actual_route.data()));
    task_data_mpi->outputs_count.emplace_back(output_data.size());
    task_data_mpi->outputs_count.emplace_back(actual_route.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    actual_route = komshina_d_grid_torus_mpi::TestTaskMPI::CalculateRoute(input_data[1], std::sqrt(world.size()),
                                                                           std::sqrt(world.size()));
    ASSERT_EQ(output_data[0], input_data[0]);
    ASSERT_EQ(actual_route, expected_route);
  }
}

TEST(komshina_d_grid_torus_mpi, SmallestGrid2Nodes) {
  boost::mpi::communicator world;
  if (world.size() != 2) {
    GTEST_SKIP();
    return;
  }

  std::vector<int> input_data{42, 1};
  std::vector<int> expected_route{0, 1};
  std::vector<int> output_data(1, 0);
  std::vector<int> actual_route;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(input_data.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(actual_route.data()));
    task_data_mpi->outputs_count.emplace_back(output_data.size());
    task_data_mpi->outputs_count.emplace_back(expected_route.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    actual_route = komshina_d_grid_torus_mpi::TestTaskMPI::CalculateRoute(input_data[1], 2, 1);
    ASSERT_EQ(output_data[0], input_data[0]);
    ASSERT_EQ(actual_route, expected_route);
  }
}

TEST(komshina_d_grid_torus_mpi, MaxRankMessagePassing) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  int max_rank = world.size() - 1;
  std::vector<int> input_data{2024, max_rank};
  std::vector<int> output_data(1, 0);
  std::vector<int> actual_route;

  size_t route_size = std::sqrt(world.size());
  actual_route.reserve(route_size);
  for (size_t i = 0; i < route_size; ++i) {
    actual_route.push_back(-1);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(input_data.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(actual_route.data()));
    task_data_mpi->outputs_count.emplace_back(output_data.size());
    task_data_mpi->outputs_count.emplace_back(actual_route.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    actual_route = komshina_d_grid_torus_mpi::TestTaskMPI::CalculateRoute(input_data[1], std::sqrt(world.size()),
                                                                          std::sqrt(world.size()));
    ASSERT_EQ(output_data[0], input_data[0]);
  }
}

TEST(komshina_d_grid_torus_mpi, WrapAroundRoute) {
  boost::mpi::communicator world;
  int size_x = static_cast<int>(std::sqrt(world.size()));
  if (world.size() < 9 || size_x * size_x != world.size()) {
    GTEST_SKIP();
    return;
  }

  std::vector<int> input_data{99, size_x - 1};
  std::vector<int> expected_route{0, size_x - 1};
  std::vector<int> output_data(1, 0);
  std::vector<int> actual_route;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(input_data.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(actual_route.data()));
    task_data_mpi->outputs_count.emplace_back(output_data.size());
    task_data_mpi->outputs_count.emplace_back(expected_route.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    actual_route = komshina_d_grid_torus_mpi::TestTaskMPI::CalculateRoute(input_data[1], size_x, size_x);
    ASSERT_EQ(output_data[0], input_data[0]);
    ASSERT_EQ(actual_route, expected_route);
  }
}

TEST(komshina_d_grid_torus_mpi, LargeDataTransfer) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  std::vector<int> input_data(10000, 42);
  std::vector<int> output_data(10000, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(input_data.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data_mpi->outputs_count.emplace_back(output_data.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data, input_data);
  }
}

TEST(komshina_d_grid_torus_mpi, RandomNodeMessagePassing) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  int random_target = world.size() / 2;
  std::vector<int> input_data{888, random_target};
  std::vector<int> output_data(1, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data_mpi->inputs_count.emplace_back(input_data.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data_mpi->outputs_count.emplace_back(output_data.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data[0], input_data[0]);
  }
}