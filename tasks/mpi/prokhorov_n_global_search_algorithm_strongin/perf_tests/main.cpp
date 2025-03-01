// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> global_a = {-1000.0};
  std::vector<double> global_b = {1000.0};
  std::vector<double> global_epsilon = {0.00001};
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

  auto test_mpi_task_parallel =
      std::make_shared<prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI>(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel->Validation(), true);
  test_mpi_task_parallel->PreProcessing();
  test_mpi_task_parallel->Run();
  test_mpi_task_parallel->PostProcessing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 100;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    EXPECT_NEAR(global_result[0], 0.0, 0.001);
  }
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> global_a = {-1000.0};
  std::vector<double> global_b = {1000.0};
  std::vector<double> global_epsilon = {0.00001};
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

  auto test_mpi_task_parallel =
      std::make_shared<prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI>(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel->Validation(), true);
  test_mpi_task_parallel->PreProcessing();
  test_mpi_task_parallel->Run();
  test_mpi_task_parallel->PostProcessing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 100;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    EXPECT_NEAR(global_result[0], 0.0, 0.001);
  }
}
