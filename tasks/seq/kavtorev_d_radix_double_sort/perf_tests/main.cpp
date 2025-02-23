#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kavtorev_d_radix_double_sort/include/ops_seq.hpp"

using kavtorev_d_radix_double_sort::RadixSortSequential;

TEST(kavtorev_d_radix_double_sort_seq, test_pipeline_run) {
  int N = 10000000;
  std::vector<double> inputData(N);
  std::vector<double> outputData(N, 0.0);

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);
    for (int i = 0; i < N; ++i) {
      inputData[i] = dist(gen);
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  task_data_seq->inputs_count.emplace_back(N);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
  task_data_seq->outputs_count.emplace_back(N);

  auto test_task_sequential = std::make_shared<kavtorev_d_radix_double_sort::RadixSortSequential>(task_data_seq);
  ASSERT_TRUE(test_task_sequential->ValidationImpl());
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perfResults);

  ppc::core::Perf::PrintPerfStatistic(perfResults);

  std::vector<double> refData = inputData;
  std::sort(refData.begin(), refData.end());
  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(refData[i], outputData[i], 1e-12);
  }
}

TEST(kavtorev_d_radix_double_sort_seq, test_task_run) {
  int N = 1000000;
  std::vector<double> inputData(N);
  std::vector<double> outputData(N, 0.0);

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);
    for (int i = 0; i < N; ++i) {
      inputData[i] = dist(gen);
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  task_data_seq->inputs_count.emplace_back(N);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
  task_data_seq->outputs_count.emplace_back(N);

  auto test_task_sequential = std::make_shared<kavtorev_d_radix_double_sort::RadixSortSequential>(task_data_seq);
  ASSERT_TRUE(test_task_sequential->ValidationImpl());
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perfResults);

  ppc::core::Perf::PrintPerfStatistic(perfResults);

  std::vector<double> refData = inputData;
  std::sort(refData.begin(), refData.end());
  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(refData[i], outputData[i], 1e-12);
  }
}