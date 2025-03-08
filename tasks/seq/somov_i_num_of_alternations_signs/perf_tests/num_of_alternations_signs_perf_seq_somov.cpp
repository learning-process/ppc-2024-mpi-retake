#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/somov_i_num_of_alternations_signs/include/num_of_alternations_signs_header_seq_somov.hpp"

TEST(somov_i_num_of_alternations_signs_seq, test_pipeline_run) {
  const int n = 100000000;
  // Create data
  std::vector<int> arr(n);
  int out = 0;
  somov_i_num_of_alternations_signs_seq::GetRndVector(arr);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  task_data->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_sequential =
      std::make_shared<somov_i_num_of_alternations_signs_seq::NumOfAlternationsSigns>(task_data);

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
  int checker = 0;
  somov_i_num_of_alternations_signs_seq::CheckForAlternationSigns(arr, checker);
  ASSERT_EQ(out, checker);
}

TEST(somov_i_num_of_alternations_signs_seq, test_task_run) {
  const int n = 100000000;
  // Create data
  std::vector<int> arr(n);
  int out = 0;
  somov_i_num_of_alternations_signs_seq::GetRndVector(arr);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  task_data->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_sequential =
      std::make_shared<somov_i_num_of_alternations_signs_seq::NumOfAlternationsSigns>(task_data);

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
  int checker = 0;
  somov_i_num_of_alternations_signs_seq::CheckForAlternationSigns(arr, checker);
  ASSERT_EQ(out, checker);
}
