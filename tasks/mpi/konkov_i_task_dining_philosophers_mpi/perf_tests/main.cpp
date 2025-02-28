#include <gtest/gtest.h>

#include <chrono>
#include <memory>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

TEST(konkov_i_task_dining_philosophers_mpi, test_pipeline_run_mpi) {
  constexpr int kNumPhilosophers = 10;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(kNumPhilosophers);

  auto task = std::make_shared<konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(konkov_i_task_dining_philosophers_mpi, test_task_run_mpi) {
  constexpr int kNumPhilosophers = 10;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(kNumPhilosophers);

  auto task = std::make_shared<konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}
