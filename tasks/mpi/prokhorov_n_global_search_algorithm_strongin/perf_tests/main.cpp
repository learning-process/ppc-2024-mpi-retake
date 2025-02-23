// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_pipeline_run) {
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

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->Validation(), true);
  testMpiTaskParallel->PreProcessing();
  testMpiTaskParallel->Run();
  testMpiTaskParallel->PostProcessing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    EXPECT_NEAR(global_result[0], 0.0, 0.001);
  }
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_task_run) {
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

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->Validation(), true);
  testMpiTaskParallel->PreProcessing();
  testMpiTaskParallel->Run();
  testMpiTaskParallel->PostProcessing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->TaskRun(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    EXPECT_NEAR(global_result[0], 0.0, 0.001);
  }
}