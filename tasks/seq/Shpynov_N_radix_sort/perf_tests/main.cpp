#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/Shpynov_N_radix_sort/include/Shpynov_N_radix_sort.hpp"

TEST(shpynov_n_radix_sort_seq, test_pipeline_run) {
  constexpr int kCount = 10;
  std::vector<int> inputVec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> expected_result = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < kCount; i++) {
    inputVec.insert(inputVec.end(), inputVec.begin(), inputVec.end());
    expected_result.insert(expected_result.end(), expected_result.begin(), expected_result.end());
  }

  std::vector<int> returned_result(inputVec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVec.data()));
  task_data_seq->inputs_count.emplace_back(inputVec.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_seq->outputs_count.emplace_back(returned_result.size());

  auto test_task_seq = std::make_shared<shpynov_n_radix_sort_seq::TestTaskSEQ>(task_data_seq);
  shpynov_n_radix_sort_seq::TestTaskSEQ test_task_seq_1(task_data_seq);
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

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_seq);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer

  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(expected_result, returned_result);
}

TEST(shpynov_n_radix_sort_seq, test_task_run) {
  constexpr int kCount = 10;
  std::vector<int> inputVec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> expected_result = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < kCount; i++) {
    inputVec.insert(inputVec.end(), inputVec.begin(), inputVec.end());
    expected_result.insert(expected_result.end(), expected_result.begin(), expected_result.end());
  }
  std::vector<int> returned_result(inputVec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVec.data()));
  task_data_seq->inputs_count.emplace_back(inputVec.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_seq->outputs_count.emplace_back(returned_result.size());

  auto test_task_seq = std::make_shared<shpynov_n_radix_sort_seq::TestTaskSEQ>(task_data_seq);
  shpynov_n_radix_sort_seq::TestTaskSEQ test_task_seq_1(task_data_seq);
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

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_seq);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  // Create Perf analyzer

  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(expected_result, returned_result);
}