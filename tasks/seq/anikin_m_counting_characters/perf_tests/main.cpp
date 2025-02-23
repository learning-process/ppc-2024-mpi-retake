// Anikin Maksim 2025
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/anikin_m_counting_characters/include/ops_seq.hpp"

TEST(anikin_m_counting_characters_seq, test_pipeline_run) {
  constexpr int kCount = 500;
  // Create data
  std::vector<char> in1;
  anikin_m_counting_characters_seq::CreateRanddataVector(&in1, kCount);
  std::vector<char> in2;
  anikin_m_counting_characters_seq::CreateRanddataVector(&in2, kCount);
  int res_out = 0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_out));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_sequential = std::make_shared<anikin_m_counting_characters_seq::TestTaskSequential>(task_data_seq);
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
  ASSERT_EQ(true, true);
}

TEST(anikin_m_counting_characters_seq, test_task_run) {
  constexpr int kCount = 500;
  // Create data
  std::vector<char> in1;
  anikin_m_counting_characters_seq::CreateRanddataVector(&in1, kCount);
  std::vector<char> in2;
  anikin_m_counting_characters_seq::CreateRanddataVector(&in2, kCount);
  int res_out = 0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_out));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_sequential = std::make_shared<anikin_m_counting_characters_seq::TestTaskSequential>(task_data_seq);
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
  ASSERT_EQ(true, true);
}