#include <gtest/gtest.h>

#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/khovansky_d_num_of_alternations_signs/include/ops_mpi.hpp"

TEST(khovansky_d_num_of_alternations_signs_mpi, test_pipeline_run) {
  constexpr int kCount = 12000;

  // Create data
  std::vector<int> in(kCount * kCount, 0);
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi>(task_data_mpi);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(0, out[0]);
}

TEST(khovansky_d_num_of_alternations_signs_mpi, test_task_run) {
  constexpr int kCount = 17000;

  // Create data
  std::vector<int> in(kCount * kCount, 0);
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi>(task_data_mpi);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(0, out[0]);
}
