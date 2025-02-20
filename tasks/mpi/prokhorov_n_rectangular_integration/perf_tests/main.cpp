#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/prokhorov_n_rectangular_integration/include/ops_mpi.hpp"

TEST(prokhorov_n_rectangular_integration_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  std::vector<double> global_input = {0.0, M_PI / 2.0, 1000000.0};
  std::vector<double> global_result(1, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
  task_data_mpi->inputs_count.emplace_back(global_input.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  task_data_mpi->outputs_count.emplace_back(global_result.size());

  auto test_task_mpi = std::make_shared<prokhorov_n_rectangular_integration_mpi::TestTaskMPI>(task_data_mpi);
  test_task_mpi->SetFunction([](double x) { return std::cos(x); });

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

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(prokhorov_n_rectangular_integration_mpi, test_task_run) {
  boost::mpi::communicator world;

  std::vector<double> global_input = {0.0, M_PI / 2.0, 1000000.0};
  std::vector<double> global_result(1, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
  task_data_mpi->inputs_count.emplace_back(global_input.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  task_data_mpi->outputs_count.emplace_back(global_result.size());

  auto test_task_mpi = std::make_shared<prokhorov_n_rectangular_integration_mpi::TestTaskMPI>(task_data_mpi);
  test_task_mpi->SetFunction([](double x) { return std::cos(x); });

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

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}