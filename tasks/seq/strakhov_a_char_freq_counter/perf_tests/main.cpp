#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/strakhov_a_char_freq_counter/include/ops_seq.hpp"

TEST(strakhov_a_char_freq_counter_seq, test_pipeline_run) {
  // Create data
  int32_t expectation = 1000;
  std::vector<char> in_string(expectation, 'a');
  std::vector<int32_t> out(1, 0);
  std::vector<char> in_target(1, 'a');

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
  task_data->inputs_count.emplace_back(in_string.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  task_data->inputs_count.emplace_back(in_target.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task = std::make_shared<strakhov_a_char_freq_counter_seq::CharFreqCounterSeq>(task_data);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(out, expectation);
}

TEST(strakhov_a_char_freq_counter_seq, test_task_run) {
  // Create data
  int32_t expectation = 1000;
  std::vector<char> in_string(expectation, 'a');
  std::vector<int32_t> out(1, 0);
  std::vector<char> in_target(1, 'a');

  // Create task_data
  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_string.data()));
  task_data_par->inputs_count.emplace_back(in_string.size());
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  task_data_par->inputs_count.emplace_back(in_target.size());
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_par->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task = std::make_shared<strakhov_a_char_freq_counter_seq::CharFreqCounterSeq>(task_data_par);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(out, expectation);
}
