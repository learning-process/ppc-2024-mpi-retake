#define OMPI_SKIP_MPICXX

#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/muradov_k_odd_even_batcher_sort/include/ops_mpi.hpp"

TEST(muradov_k_odd_even_batcher_sort_mpi, test_pipeline_run) {
  constexpr int n = 256 * 1024;
  const int k_iterations = 100;

  std::vector<int> input(n);
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < n; ++i) {
    input[i] = std::rand() % 1000;
  }
  std::vector<int> output(n, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());
  task_data->state_of_testing = ppc::core::TaskData::kPerf;

  auto sort_task = std::make_shared<muradov_k_odd_even_batcher_sort::OddEvenBatcherSortTask>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(sort_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    std::vector<int> expected = input;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(output, expected);
  }
}

TEST(muradov_k_odd_even_batcher_sort_mpi, test_task_run) {
  constexpr int n = 1024;
  std::vector<int> input(n);
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < n; ++i) {
    input[i] = std::rand() % 1000;
  }
  std::vector<int> output(n, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());
  task_data->state_of_testing = ppc::core::TaskData::kPerf;

  auto sort_task = std::make_shared<muradov_k_odd_even_batcher_sort::OddEvenBatcherSortTask>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(sort_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    std::vector<int> expected = input;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(output, expected);
  }
}