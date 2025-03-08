#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/prokhorov_n_rectangular_integration/include/ops_seq.hpp"

TEST(prokhorov_n_rectangular_integration_seq, test_pipeline_run) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n = 100000;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<prokhorov_n_rectangular_integration_seq::TestTaskSequential>(task_data_seq);
  test_task_sequential->SetFunction([](double x) { return x * x; });

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  const double expected_result = 1.0 / 3.0;
  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_task_run) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n = 100000;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential =
      std::make_shared<prokhorov_n_rectangular_integration_seq::TestTaskSequential>(task_data_seq);
  test_task_sequential->SetFunction([](double x) { return x * x; });

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  const double expected_result = 1.0 / 3.0;
  ASSERT_NEAR(out[0], expected_result, 1e-3);
}