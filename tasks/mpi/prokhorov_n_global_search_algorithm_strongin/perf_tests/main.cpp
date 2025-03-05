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

  auto test_task_mpi = std::make_shared<prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI>(
      task_data_mpi, quadratic_function);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  EXPECT_NEAR(out[0], 0.0, 0.001);
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_task_run) {
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

  auto test_task_mpi = std::make_shared<prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskMPI>(
      task_data_mpi, quadratic_function);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  EXPECT_NEAR(out[0], 0.0, 0.001);
}