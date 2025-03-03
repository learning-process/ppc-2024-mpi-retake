#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

namespace {
double ComputeValue(int i) { return (std::sin(i) * 1e9) + (std::cos(i * 0.5) * 1e8); }
}  // namespace

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_pipeline_run) {
  int size = 10000000;
  std::vector<double> in(size);
  std::vector<double> out(size, 0.0);

  in.resize(size);
  for (int i = 0; i < size; ++i) {
    in[i] = ComputeValue(i);
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(size);

  // Create Task
  auto test_task_sequential =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential>(
          task_data_seq);

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
  std::vector<double> ref_data = in;
  std::ranges::sort(ref_data);
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(ref_data[i], out[i], 1e-12);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_task_run) {
  int size = 10000000;
  std::vector<double> in(size);
  std::vector<double> out(size, 0.0);

  in.resize(size);
  for (int i = 0; i < size; ++i) {
    in[i] = ComputeValue(i);
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(size);

  // Create Task
  auto test_task_sequential =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential>(
          task_data_seq);

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
  std::vector<double> ref_data = in;
  std::ranges::sort(ref_data);
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(ref_data[i], out[i], 1e-12);
  }
}