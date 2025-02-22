#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "mpi/prokhorov_n_rectangular_integration/include/ops_mpi.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

TEST(prokhorov_n_rectangular_integration_mpi, test_integration_cos_x) {
  boost::mpi::communicator world;
  std::vector<double> global_input = {0.0, M_PI / 2.0, 1000.0};
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
  task_data_par->inputs_count.emplace_back(global_input.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  task_data_par->outputs_count.emplace_back(global_result.size());

  auto test_mpi_task_parallel = std::make_shared<prokhorov_n_rectangular_integration_mpi::TestTaskMPI>(task_data_par);
  test_mpi_task_parallel->SetFunction([](double x) { return std::cos(x); });

  ASSERT_EQ(test_mpi_task_parallel->ValidationImpl(), true);
  test_mpi_task_parallel->PreProcessingImpl();
  test_mpi_task_parallel->RunImpl();
  test_mpi_task_parallel->PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    task_data_seq->inputs_count.emplace_back(global_input.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    task_data_seq->outputs_count.emplace_back(reference_result.size());

    auto test_mpi_task_sequential =
        std::make_shared<prokhorov_n_rectangular_integration_mpi::TestTaskSequential>(task_data_seq);
    test_mpi_task_sequential->SetFunction([](double x) { return std::cos(x); });

    ASSERT_EQ(test_mpi_task_sequential->ValidationImpl(), true);
    test_mpi_task_sequential->PreProcessingImpl();
    test_mpi_task_sequential->RunImpl();
    test_mpi_task_sequential->PostProcessingImpl();

    ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
  }
}

TEST(prokhorov_n_rectangular_integration_mpi, test_integration_x_cubed) {
  boost::mpi::communicator world;
  std::vector<double> global_input = {0.0, 1.0, 1000.0};
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
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
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
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

TEST(prokhorov_n_rectangular_integration_mpi, test_integration_one_over_x) {
  boost::mpi::communicator world;
  std::vector<double> global_input = {1.0, 2.0, 1000.0};
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
  task_data_par->inputs_count.emplace_back(global_input.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  task_data_par->outputs_count.emplace_back(global_result.size());

  auto test_mpi_task_parallel = std::make_shared<prokhorov_n_rectangular_integration_mpi::TestTaskMPI>(task_data_par);
  test_mpi_task_parallel->SetFunction([](double x) { return 1.0 / x; });

  ASSERT_EQ(test_mpi_task_parallel->ValidationImpl(), true);
  test_mpi_task_parallel->PreProcessingImpl();
  test_mpi_task_parallel->RunImpl();
  test_mpi_task_parallel->PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    task_data_seq->inputs_count.emplace_back(global_input.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    task_data_seq->outputs_count.emplace_back(reference_result.size());

    auto test_mpi_task_sequential =
        std::make_shared<prokhorov_n_rectangular_integration_mpi::TestTaskSequential>(task_data_seq);
    test_mpi_task_sequential->SetFunction([](double x) { return 1.0 / x; });

    ASSERT_EQ(test_mpi_task_sequential->ValidationImpl(), true);
    test_mpi_task_sequential->PreProcessingImpl();
    test_mpi_task_sequential->RunImpl();
    test_mpi_task_sequential->PostProcessingImpl();

    ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
  }
}