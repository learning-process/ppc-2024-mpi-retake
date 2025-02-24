#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_mpi, test_pipeline_run) {
  boost::mpi::communicator world_;
  int width = static_cast<int>(std::sqrt(world_.size()));
  if (width * width != world_.size()) {
    return;
  }

  const std::string data_input(100000, 'a');
  int dest = world_.size() - 1;
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData input_data{data_input, dest};
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData output_data;

  std::vector<int> path_expected = komshina_d_grid_torus_mpi::TestTaskMPI::calculate_route(dest, world_.size(), width);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_data));
  task_data_mpi->inputs_count.emplace_back(sizeof(komshina_d_grid_torus_mpi::TestTaskMPI::InputData));
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output_data));
  task_data_mpi->outputs_count.emplace_back(sizeof(komshina_d_grid_torus_mpi::TestTaskMPI::InputData));

  auto test_task_mpi = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(task_data_mpi);

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

  if (world_.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(output_data.payload, input_data.payload);
    ASSERT_EQ(output_data.path, path_expected);
  }
}

TEST(komshina_d_grid_torus_mpi, test_task_run) {
  boost::mpi::communicator world_;
  int width = static_cast<int>(std::sqrt(world_.size()));
  if (width * width != world_.size()) {
    return;
  }

  const std::string data_input(100000, 'a');
  int dest = world_.size() - 1;
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData input_data{data_input, dest};
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData output_data;

  std::vector<int> path_expected = komshina_d_grid_torus_mpi::TestTaskMPI::calculate_route(dest, world_.size(), width);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_data));
  task_data_mpi->inputs_count.emplace_back(sizeof(komshina_d_grid_torus_mpi::TestTaskMPI::InputData));
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output_data));
  task_data_mpi->outputs_count.emplace_back(sizeof(komshina_d_grid_torus_mpi::TestTaskMPI::InputData));

  auto test_task_mpi = std::make_shared<komshina_d_grid_torus_mpi::TestTaskMPI>(task_data_mpi);

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

  if (world_.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(output_data.payload, input_data.payload);
    ASSERT_EQ(output_data.path, path_expected);
  }
}