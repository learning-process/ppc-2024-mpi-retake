// Anikin Maksim 2025
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/anikin_m_graham_scan/include/ops_seq.hpp"

namespace {
void CreateRandomData(std::vector<anikin_m_graham_scan_seq::Pt> &alg_in, int count) {
  alg_in.clear();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 100.0);
  anikin_m_graham_scan_seq::Pt rand;
  for (int i = 0; i < count; i++) {
    rand.x = dis(gen);
    rand.y = dis(gen);
    alg_in.push_back(rand);
  }
}
}  // namespace

TEST(anikin_m_graham_scan_seq, test_pipeline_run) {
  constexpr int kCount = 2000000;

  std::vector<anikin_m_graham_scan_seq::Pt> in;
  std::vector<anikin_m_graham_scan_seq::Pt> out;

  CreateRandomData(in, kCount);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());

  // Create Task
  auto test_task_sequential = std::make_shared<anikin_m_graham_scan_seq::TestTaskSequential>(task_data_seq);

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

TEST(anikin_m_graham_scan_seq, test_task_run) {
  constexpr int kCount = 2000000;

  std::vector<anikin_m_graham_scan_seq::Pt> in;
  std::vector<anikin_m_graham_scan_seq::Pt> out;

  CreateRandomData(in, kCount);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());

  // Create Task
  auto test_task_sequential = std::make_shared<anikin_m_graham_scan_seq::TestTaskSequential>(task_data_seq);

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