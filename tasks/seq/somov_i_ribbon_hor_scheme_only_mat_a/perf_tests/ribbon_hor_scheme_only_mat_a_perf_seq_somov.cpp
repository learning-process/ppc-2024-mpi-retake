#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/somov_i_ribbon_hor_scheme_only_mat_a/include/ribbon_hor_scheme_only_mat_a_header_seq_somov.hpp"

TEST(somov_i_ribbon_hor_scheme_only_mat_a_seq, test_pipeline_run) {
  // Create data
  const int a_r = 500;
  const int a_c = 500;
  const int b_r = 500;
  const int b_c = 500;
  std::vector<int> a(a_c * a_r);
  std::vector<int> b(b_r * b_c);
  std::vector<int> c(a_r * b_c);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(a);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(b);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs_count.emplace_back(a_c);
  task_data->inputs_count.emplace_back(a_r);
  task_data->inputs_count.emplace_back(b_c);
  task_data->inputs_count.emplace_back(b_r);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data->outputs_count.emplace_back(c.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<somov_i_ribbon_hor_scheme_only_mat_a_seq::RibbonHorSchemeOnlyMatA>(task_data);

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
  std::vector<int> checker(a_r * b_c, 0);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::LiterallyMult(a, b, checker, a_c, a_r, b_c);
  ASSERT_EQ(checker, c);
}

TEST(somov_i_ribbon_hor_scheme_only_mat_a_seq, test_task_run) {
  // Create data
  const int a_r = 500;
  const int a_c = 500;
  const int b_r = 500;
  const int b_c = 500;
  std::vector<int> a(a_c * a_r);
  std::vector<int> b(b_r * b_c);
  std::vector<int> c(a_r * b_c);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(a);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::GetRndVector(b);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs_count.emplace_back(a_c);
  task_data->inputs_count.emplace_back(a_r);
  task_data->inputs_count.emplace_back(b_c);
  task_data->inputs_count.emplace_back(b_r);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data->outputs_count.emplace_back(c.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<somov_i_ribbon_hor_scheme_only_mat_a_seq::RibbonHorSchemeOnlyMatA>(task_data);

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
  std::vector<int> checker(a_r * b_c, 0);
  somov_i_ribbon_hor_scheme_only_mat_a_seq::LiterallyMult(a, b, checker, a_c, a_r, b_c);
  ASSERT_EQ(checker, c);
}
