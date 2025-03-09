#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/prokhorov_n_rectangular_integration/include/ops_mpi.hpp"

TEST(prokhorov_n_rectangular_integration_mpi, test_integration_x_cubed) {
  boost::mpi::communicator world;
  std::vector<double> global_input = {0.0, 1.0, 1000.0};
  std::vector<double> global_result(1, 0.0);

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
  task_data_par->inputs_count.emplace_back(global_input.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  task_data_par->outputs_count.emplace_back(global_result.size());

  auto test_mpi_task_parallel = std::make_shared<prokhorov_n_rectangular_integration_mpi::TestTaskMPI>(task_data_par);
  test_mpi_task_parallel->SetFunction([](double x) { return x * x * x; });

  ASSERT_EQ(test_mpi_task_parallel->ValidationImpl(), true);
  test_mpi_task_parallel->PreProcessingImpl();
  test_mpi_task_parallel->RunImpl();
  test_mpi_task_parallel->PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    task_data_seq->inputs_count.emplace_back(global_input.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    task_data_seq->outputs_count.emplace_back(reference_result.size());

    auto test_mpi_task_sequential =
        std::make_shared<prokhorov_n_rectangular_integration_mpi::TestTaskSequential>(task_data_seq);
    test_mpi_task_sequential->SetFunction([](double x) { return x * x * x; });

    ASSERT_EQ(test_mpi_task_sequential->ValidationImpl(), true);
    test_mpi_task_sequential->PreProcessingImpl();
    test_mpi_task_sequential->RunImpl();
    test_mpi_task_sequential->PostProcessingImpl();

    ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
  }
}

TEST(prokhorov_n_rectangular_integration_mpi, test_integration_x_squared) {
  boost::mpi::communicator world;
  std::vector<double> global_input = {0.0, 1.0, 1000.0};
  std::vector<double> global_result(1, 0.0);

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
  task_data_par->inputs_count.emplace_back(global_input.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  task_data_par->outputs_count.emplace_back(global_result.size());

  auto test_mpi_task_parallel = std::make_shared<prokhorov_n_rectangular_integration_mpi::TestTaskMPI>(task_data_par);
  test_mpi_task_parallel->SetFunction([](double x) { return x * x; });

  ASSERT_TRUE(test_mpi_task_parallel->ValidationImpl());

  test_mpi_task_parallel->PreProcessingImpl();
  test_mpi_task_parallel->RunImpl();
  test_mpi_task_parallel->PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    task_data_seq->inputs_count.emplace_back(global_input.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    task_data_seq->outputs_count.emplace_back(reference_result.size());

    auto test_mpi_task_sequential =
        std::make_shared<prokhorov_n_rectangular_integration_mpi::TestTaskSequential>(task_data_seq);
    test_mpi_task_sequential->SetFunction([](double x) { return x * x; });

    ASSERT_TRUE(test_mpi_task_sequential->ValidationImpl());

    test_mpi_task_sequential->PreProcessingImpl();
    test_mpi_task_sequential->RunImpl();
    test_mpi_task_sequential->PostProcessingImpl();

    double a = global_input[0];
    double b = global_input[1];
    double expected_result = (b * b * b / 3.0) - (a * a * a / 3.0);

    ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
    ASSERT_NEAR(global_result[0], expected_result, 1e-3);
  }
}