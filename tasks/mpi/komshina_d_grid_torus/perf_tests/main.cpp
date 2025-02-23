#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <memory>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

TEST(komshina_d_grid_torus_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    return;
  }

  const std::string data_input(16381, 'a');
  int dest = world.size() - 1;
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData in(data_input, dest);
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData out;

  int size = static_cast<int>(std::sqrt(world.size()));
  std::vector<int> route_expected =
      komshina_d_grid_torus_mpi::TestTaskMPI::CalculateRoute(dest, size, size);

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&in));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
    task_data_mpi->outputs_count.emplace_back(1);
  }

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

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(out.payload, in.payload);
    ASSERT_EQ(out.path, route_expected);
  }
}

TEST(komshina_d_grid_torus_mpi, test_task_run) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    return;
  }

  const std::string data_input(16381, 'a');
  int dest = world.size() - 1;
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData in(data_input, dest);
  komshina_d_grid_torus_mpi::TestTaskMPI::InputData out;

  int size = static_cast<int>(std::sqrt(world.size()));
  std::vector<int> route_expected =
      komshina_d_grid_torus_mpi::TestTaskMPI::CalculateRoute(dest, size, size);


  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&in));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
    task_data_mpi->outputs_count.emplace_back(1);
  }

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

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(out.payload, in.payload);
    ASSERT_EQ(out.path, route_expected);
  }
}
