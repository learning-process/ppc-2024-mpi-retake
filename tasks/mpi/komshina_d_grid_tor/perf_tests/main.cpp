#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/komshina_d_grid_tor/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    return;
  }

  const std::string data_input(100000, 'a');
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(data_input.data())));
    task_data_mpi->inputs_count.emplace_back(data_input.size());
    task_data_mpi->outputs.emplace_back(new uint8_t[data_input.size()]);
    task_data_mpi->outputs_count.emplace_back(data_input.size());
  }

  auto test_task_mpi = std::make_shared<komshina_d_grid_torus_topology_mpi::TestTaskMPI>(task_data_mpi);

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
    ASSERT_EQ(memcmp(task_data_mpi->inputs[0], task_data_mpi->outputs[0], data_input.size()), 0);
  }
}

TEST(komshina_d_grid_torus_mpi, test_task_run) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    return;
  }

  const std::string data_input(100000, 'a');
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(data_input.data())));
    task_data_mpi->inputs_count.emplace_back(data_input.size());
    task_data_mpi->outputs.emplace_back(new uint8_t[data_input.size()]);
    task_data_mpi->outputs_count.emplace_back(data_input.size());
  }

  auto test_task_mpi = std::make_shared<komshina_d_grid_torus_topology_mpi::TestTaskMPI>(task_data_mpi);

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
    ASSERT_EQ(memcmp(task_data_mpi->inputs[0], task_data_mpi->outputs[0], data_input.size()), 0);
  }
}