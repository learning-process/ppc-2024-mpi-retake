#define OMPI_SKIP_MPICXX
#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/muradov_k_radix_sort/include/ops_mpi.hpp"

TEST(muradov_k_radix_sort_mpi, test_pipeline_run) {
  constexpr int kN = 256 * 1024;
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  std::vector<int> input(kN);
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < kN; ++i) {
    input[i] = std::rand() % 1000;
  }
  std::vector<int> output(input.size(), 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());
  task_data->state_of_testing = ppc::core::TaskData::kPerf;
  auto sort_task = std::make_shared<muradov_k_radix_sort::RadixSortTask>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 100;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(sort_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (proc_rank == 0) {
    std::vector<int> expected = input;
    std::ranges::sort(expected);
    ASSERT_EQ(output, expected);
  }
}

TEST(muradov_k_radix_sort_mpi, test_task_run) {
  constexpr int kN = 20480;
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  std::vector<int> input(kN);
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < kN; ++i) {
    input[i] = std::rand() % 1000;
  }
  std::vector<int> output(input.size(), 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());
  task_data->state_of_testing = ppc::core::TaskData::kPerf;
  auto sort_task = std::make_shared<muradov_k_radix_sort::RadixSortTask>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 300;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(sort_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (proc_rank == 0) {
    std::vector<int> expected = input;
    std::ranges::sort(expected);
    ASSERT_EQ(output, expected);
  }
}