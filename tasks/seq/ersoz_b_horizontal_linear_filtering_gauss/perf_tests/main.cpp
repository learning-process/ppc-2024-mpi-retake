#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/ersoz_b_horizontal_linear_filtering_gauss/include/ops_seq.hpp"

TEST(ersoz_b_test_task_seq, test_pipeline_run) {
  constexpr int kN = 128;
  std::vector<char> in(kN * kN, 0);
  for (int i = 0; i < kN; i++) {
    for (int j = 0; j < kN; j++) {
      in[(i * kN) + j] = static_cast<char>((i + j) % 256);
    }
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.push_back(in.size());
  std::vector<char> out((kN - 2) * (kN - 2), 0);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.push_back(out.size());

  auto task = std::make_shared<ersoz_b_test_task_seq::TestTaskSequential>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  task->PostProcessing();
}

TEST(ersoz_b_test_task_seq, test_task_run) {
  constexpr int kN = 128;
  std::vector<char> in(kN * kN, 0);
  for (int i = 0; i < kN; i++) {
    for (int j = 0; j < kN; j++) {
      in[(i * kN) + j] = static_cast<char>((i + j) % 256);
    }
  }

  std::vector<char> out((kN - 2) * (kN - 2), 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.push_back(in.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.push_back(out.size());

  auto task = std::make_shared<ersoz_b_test_task_seq::TestTaskSequential>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  task->PostProcessing();
}
