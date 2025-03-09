// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/shishkarev_a_dijkstra_algorithm/include/ops_seq.hpp"

TEST(shishkarev_a_dijkstra_algorithm_seq, test_PipelineRun) {
  int count_size_vector = 5000;
  int st = 0;
  std::vector<int> global_matrix(count_size_vector * count_size_vector, 1);
  std::vector<int32_t> global_path(count_size_vector, 1);

  for (int i = 0; i < count_size_vector; i++) {
    global_matrix[(i * count_size_vector) + i] = 0;
  }
  global_path[0] = 0;

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
  task_data_seq->inputs_count.emplace_back(global_matrix.size());
  task_data_seq->inputs_count.emplace_back(count_size_vector);
  task_data_seq->inputs_count.emplace_back(st);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_path.data()));
  task_data_seq->outputs_count.emplace_back(global_path.size());

  auto test_task_sequential = std::make_shared<shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential>(task_data_seq);

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
  ASSERT_EQ(1, global_path[1]);
}

TEST(shishkarev_a_dijkstra_algorithm_seq, test_task_run) {
  int count_size_vector = 5000;
  int st = 0;
  std::vector<int> global_matrix(count_size_vector * count_size_vector, 1);
  std::vector<int32_t> global_path(count_size_vector, 1);

  for (int i = 0; i < count_size_vector; i++) {
    global_matrix[(i * count_size_vector) + i] = 0;
  }
  global_path[0] = 0;

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
  task_data_seq->inputs_count.emplace_back(global_matrix.size());
  task_data_seq->inputs_count.emplace_back(count_size_vector);
  task_data_seq->inputs_count.emplace_back(st);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_path.data()));
  task_data_seq->outputs_count.emplace_back(global_path.size());

  auto test_task_sequential = std::make_shared<shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential>(task_data_seq);

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
  ASSERT_EQ(1, global_path[1]);
}