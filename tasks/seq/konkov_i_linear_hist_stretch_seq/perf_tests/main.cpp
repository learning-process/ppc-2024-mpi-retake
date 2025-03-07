#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/konkov_i_linear_hist_stretch_seq/include/ops_seq.hpp"

TEST(konkov_i_linear_hist_stretch_seq, test_pipeline_run) {
  constexpr size_t kSize = 100000000;

  std::vector<uint8_t> input(kSize, 127);
  input.front() = 0;
  input.back() = 255;
  std::vector<uint8_t> output(kSize, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_seq->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<konkov_i_linear_hist_stretch_seq::LinearHistStretchSeq>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(output.front(), 0);
  ASSERT_EQ(output.back(), 255);
}

TEST(konkov_i_linear_hist_stretch_seq, test_task_run) {
  constexpr size_t kSize = 100000000;

  std::vector<uint8_t> input(kSize, 127);
  input.front() = 0;
  input.back() = 255;
  std::vector<uint8_t> output(kSize, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(input.data());
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(output.data());
  task_data_seq->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<konkov_i_linear_hist_stretch_seq::LinearHistStretchSeq>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(output.front(), 0);
  ASSERT_EQ(output.back(), 255);
}
