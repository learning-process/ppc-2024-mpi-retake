#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/vasenkov_a_gauss_jordan/include/ops_seq.hpp"

TEST(vasenkov_a_gauss_jordan_seq, test_pipeline_run) {
  std::vector<double> input_matrix = {1, 0, 0, 5, 0, 1, 0, -3, 0, 0, 1, 2};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::vector<double> expected_result = {1, 0, 0, 5, 0, 1, 0, -3, 0, 0, 1, 2};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(input_matrix.data())));
  task_data_seq->inputs_count.emplace_back(input_matrix.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  // Create Task
  auto test_task_sequential = std::make_shared<vasenkov_a_gauss_jordan_seq::GaussJordanMethodSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10000;
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
  ASSERT_EQ(output_result, expected_result);
}

TEST(vasenkov_a_gauss_jordan_seq, test_task_run) {
  std::vector<double> input_matrix = {1, 0, 0, 5, 0, 1, 0, -3, 0, 0, 1, 2};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::vector<double> expected_result = {1, 0, 0, 5, 0, 1, 0, -3, 0, 0, 1, 2};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(input_matrix.data())));
  task_data_seq->inputs_count.emplace_back(input_matrix.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  // Create Task
  auto test_task_sequential = std::make_shared<vasenkov_a_gauss_jordan_seq::GaussJordanMethodSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 100000000;
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
  ASSERT_EQ(output_result, expected_result);
}
