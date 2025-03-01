// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_strongin_algorithm_quadratic_function) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<double> global_a = {-10.0};
  std::vector<double> global_b = {10.0};
  std::vector<double> global_epsilon = {0.001};
  std::vector<double> global_result(1, 0.0);

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
  task_data_par->inputs_count.emplace_back(global_a.size());
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
  task_data_par->inputs_count.emplace_back(global_b.size());
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
  task_data_par->inputs_count.emplace_back(global_epsilon.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  task_data_par->outputs_count.emplace_back(global_result.size());

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI test_task_mpi(task_data_par);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
    task_data_seq->inputs_count.emplace_back(global_a.size());
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
    task_data_seq->inputs_count.emplace_back(global_b.size());
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
    task_data_seq->inputs_count.emplace_back(global_epsilon.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    task_data_seq->outputs_count.emplace_back(reference_result.size());

    prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.Validation(), true);
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();

    EXPECT_NEAR(reference_result[0], global_result[0], 0.001);
  }

  world.barrier();
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_strongin_algorithm_sinus_function) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<double> global_a = {0.0};
  std::vector<double> global_b = {3.14};
  std::vector<double> global_epsilon = {0.001};
  std::vector<double> global_result(1, 0.0);

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
  task_data_par->inputs_count.emplace_back(global_a.size());
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
  task_data_par->inputs_count.emplace_back(global_b.size());
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
  task_data_par->inputs_count.emplace_back(global_epsilon.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  task_data_par->outputs_count.emplace_back(global_result.size());

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI test_task_mpi(task_data_par);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
    task_data_seq->inputs_count.emplace_back(global_a.size());
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
    task_data_seq->inputs_count.emplace_back(global_b.size());
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
    task_data_seq->inputs_count.emplace_back(global_epsilon.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    task_data_seq->outputs_count.emplace_back(reference_result.size());

    prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.Validation(), true);
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();

    EXPECT_NEAR(reference_result[0], global_result[0], 0.001);
  }

  world.barrier();
}