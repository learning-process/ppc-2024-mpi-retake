#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/vasenkov_a_bellman_ford/include/ops_seq.hpp"

TEST(vasenkov_a_bellman_ford_seq, test_pipeline_run) {
  // Create data
  std::vector<int> row_ptr = {0, 2, 4, 5, 5};
  std::vector<int> col_ind = {1, 2, 2, 3, 3};
  std::vector<int> weights = {4, 5, -3, 2, 1};
  int num_vertices = 4;
  int source_vertex = 0;

  std::vector<int> expected_distances = {0, 4, 1, 2};
  std::vector<int> global_result(num_vertices);

  // Create task_data

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(row_ptr.data())));
  task_data_par->inputs_count.emplace_back(row_ptr.size());

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(col_ind.data())));
  task_data_par->inputs_count.emplace_back(col_ind.size());

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(weights.data())));
  task_data_par->inputs_count.emplace_back(weights.size());

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&num_vertices)));
  task_data_par->inputs_count.emplace_back(1);

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&source_vertex)));
  task_data_par->inputs_count.emplace_back(1);

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
  task_data_par->outputs_count.emplace_back(global_result.size());

  // Create Task
  auto test_task_sequential = std::make_shared<vasenkov_a_bellman_ford_seq::BellmanFordSequential>(task_data_par);

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
  ASSERT_EQ(expected_distances, global_result);
}

TEST(vasenkov_a_bellman_ford_seq, test_task_run) {
  // Create data
  std::vector<int> row_ptr = {0, 2, 4, 5, 5};
  std::vector<int> col_ind = {1, 2, 2, 3, 3};
  std::vector<int> weights = {4, 5, -3, 2, 1};
  int num_vertices = 4;
  int source_vertex = 0;

  std::vector<int> expected_distances = {0, 4, 1, 2};
  std::vector<int> global_result(num_vertices);

  // Create task_data

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(row_ptr.data())));
  task_data_par->inputs_count.emplace_back(row_ptr.size());

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(col_ind.data())));
  task_data_par->inputs_count.emplace_back(col_ind.size());

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(weights.data())));
  task_data_par->inputs_count.emplace_back(weights.size());

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&num_vertices)));
  task_data_par->inputs_count.emplace_back(1);

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&source_vertex)));
  task_data_par->inputs_count.emplace_back(1);

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
  task_data_par->outputs_count.emplace_back(global_result.size());

  // Create Task
  auto test_task_sequential = std::make_shared<vasenkov_a_bellman_ford_seq::BellmanFordSequential>(task_data_par);

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
  ASSERT_EQ(expected_distances, global_result);
}
