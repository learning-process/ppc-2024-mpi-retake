// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
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

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
  taskDataPar->inputs_count.emplace_back(global_a.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
  taskDataPar->inputs_count.emplace_back(global_b.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
  taskDataPar->inputs_count.emplace_back(global_epsilon.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataPar->outputs_count.emplace_back(global_result.size());

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI testTaskMPI(taskDataPar);
  ASSERT_EQ(testTaskMPI.Validation(), true);
  testTaskMPI.PreProcessing();
  testTaskMPI.Run();
  testTaskMPI.PostProcessing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
    taskDataSeq->inputs_count.emplace_back(global_a.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
    taskDataSeq->inputs_count.emplace_back(global_b.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
    taskDataSeq->inputs_count.emplace_back(global_epsilon.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.Validation(), true);
    testTaskSequential.PreProcessing();
    testTaskSequential.Run();
    testTaskSequential.PostProcessing();

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

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
  taskDataPar->inputs_count.emplace_back(global_a.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
  taskDataPar->inputs_count.emplace_back(global_b.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
  taskDataPar->inputs_count.emplace_back(global_epsilon.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataPar->outputs_count.emplace_back(global_result.size());

  prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI testTaskMPI(taskDataPar);
  ASSERT_EQ(testTaskMPI.Validation(), true);
  testTaskMPI.PreProcessing();
  testTaskMPI.Run();
  testTaskMPI.PostProcessing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
    taskDataSeq->inputs_count.emplace_back(global_a.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
    taskDataSeq->inputs_count.emplace_back(global_b.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
    taskDataSeq->inputs_count.emplace_back(global_epsilon.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.Validation(), true);
    testTaskSequential.PreProcessing();
    testTaskSequential.Run();
    testTaskSequential.PostProcessing();

    EXPECT_NEAR(reference_result[0], global_result[0], 0.001);
  }

  world.barrier();
}