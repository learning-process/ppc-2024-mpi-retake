// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, Test_Strongin_Algorithm_Quadratic_Function) {
  std::vector<double> in_a = {-10.0};
  std::vector<double> in_b = {10.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  task_data_mpi->inputs_count.emplace_back(in_a.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  task_data_mpi->inputs_count.emplace_back(in_b.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  task_data_mpi->inputs_count.emplace_back(in_epsilon.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  auto quadratic_function = [](double x) { return x * x; };

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI test_task_mpi(task_data_mpi, quadratic_function);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  EXPECT_NEAR(out[0], 0.0, 0.001);
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, Test_Strongin_Algorithm_Absolute_Function) {
  std::vector<double> in_a = {-5.0};
  std::vector<double> in_b = {5.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  task_data_mpi->inputs_count.emplace_back(in_a.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  task_data_mpi->inputs_count.emplace_back(in_b.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  task_data_mpi->inputs_count.emplace_back(in_epsilon.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  auto absolute_function = [](double x) { return std::abs(x); };

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI test_task_mpi(task_data_mpi, absolute_function);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  EXPECT_NEAR(out[0], 0.0, 0.001);
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, Test_Strongin_Algorithm_SquareRoot_Function) {
  std::vector<double> in_a = {0.0};
  std::vector<double> in_b = {10.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  task_data_mpi->inputs_count.emplace_back(in_a.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  task_data_mpi->inputs_count.emplace_back(in_b.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  task_data_mpi->inputs_count.emplace_back(in_epsilon.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  auto square_root_function = [](double x) { return std::sqrt(x); };

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI test_task_mpi(task_data_mpi, square_root_function);
  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  EXPECT_NEAR(out[0], 0.0, 0.001);
}
