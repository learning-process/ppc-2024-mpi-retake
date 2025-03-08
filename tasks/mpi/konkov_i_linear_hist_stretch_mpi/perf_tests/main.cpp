#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/konkov_i_linear_hist_stretch_mpi/include/ops_mpi.hpp"

TEST(konkov_i_linear_hist_stretch_mpi, test_pipeline_run_mpi) {
  constexpr size_t kSize = 500000000;
  std::vector<uint8_t> in(kSize, 128);
  std::vector<uint8_t> out(kSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(konkov_i_linear_hist_stretch_mpi, test_task_run_mpi) {
  constexpr size_t kSize = 1000000000;
  std::vector<uint8_t> in(kSize, 128);
  std::vector<uint8_t> out(kSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = std::make_shared<konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}
