#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/makhov_m_monte_carlo_method/include/ops_seq.hpp"

TEST(makhov_m_monte_carlo_method_seq, test_pipeline_run) {
  // Create data
  std::function<double(const std::vector<double> &)> f = [](const std::vector<double> &x) {
    return (x[0] * x[0]) + (x[1] * x[1]);
  };
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}};
  double *answer_ptr = nullptr;
  double reference = 0.67;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(2);  // Integral dimension info
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_sequential = std::make_shared<makhov_m_monte_carlo_method_seq::TestTaskSequential>(task_data_seq);

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
  uint8_t *answer_data = task_data_seq->outputs[0];
  double retrieved_value = NAN;
  std::memcpy(&retrieved_value, answer_data, sizeof(double));
  double truncated_value = std::round(retrieved_value * 100) / 100;
  ASSERT_EQ(reference, truncated_value);
}

TEST(makhov_m_monte_carlo_method_seq, test_task_run) {
  // Create data
  std::function<double(const std::vector<double> &)> f = [](const std::vector<double> &x) {
    return (x[0] * x[0]) + (x[1] * x[1]);
  };
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}};
  double *answer_ptr = nullptr;
  double reference = 0.67;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->inputs_count.emplace_back(2);  // Integral dimension info
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_sequential = std::make_shared<makhov_m_monte_carlo_method_seq::TestTaskSequential>(task_data_seq);

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
  uint8_t *answer_data = task_data_seq->outputs[0];
  double retrieved_value = NAN;
  std::memcpy(&retrieved_value, answer_data, sizeof(double));
  double truncated_value = std::round(retrieved_value * 100) / 100;
  ASSERT_EQ(reference, truncated_value);
}
