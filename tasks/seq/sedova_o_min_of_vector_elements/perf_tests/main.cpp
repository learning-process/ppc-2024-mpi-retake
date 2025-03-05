#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sedova_o_min_of_vector_elements/include/ops_seq.hpp"

TEST(sedova_o_min_of_vector_elements_seq, test_pipeline_run) {
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  int ref = INT_MIN;

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  int rows = 10000;
  int columns = 10000;
  int min = -500;
  int max = 500;

  global_matrix = sedova_o_min_of_vector_elements_seq::GetRandomMatrix(rows, columns, min, max);
  int index = (static_cast<int>(gen() % (rows * columns)));
  global_matrix[index / columns][index / rows] = ref;

  for (unsigned int i = 0; i < global_matrix.size(); i++) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    task_data_seq->inputs_count.emplace_back(rows);
    task_data_seq->inputs_count.emplace_back(columns);

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_min.data()));
    task_data_seq->outputs_count.emplace_back(global_min.size());
  }

  // Create Task
  auto test_task_sequential = std::make_shared<sedova_o_min_of_vector_elements_seq::TestTaskSequential>(task_data_seq);

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
  ASSERT_EQ(ref, global_min[0]);
}

TEST(sedova_o_min_of_vector_elements_seq, test_task_run) {
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  int ref = INT_MIN;

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  int rows = 10000;
  int columns = 10000;
  int min = -500;
  int max = 500;

  global_matrix = sedova_o_min_of_vector_elements_seq::GetRandomMatrix(rows, columns, min, max);
  int index = (static_cast<int>(gen() % (rows * columns)));
  global_matrix[index / columns][index / rows] = ref;

  for (unsigned int i = 0; i < global_matrix.size(); i++) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    task_data_seq->inputs_count.emplace_back(rows);
    task_data_seq->inputs_count.emplace_back(columns);

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_min.data()));
    task_data_seq->outputs_count.emplace_back(global_min.size());
  }

  // Create Task
  auto test_task_sequential = std::make_shared<sedova_o_min_of_vector_elements_seq::TestTaskSequential>(task_data_seq);

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
  ASSERT_EQ(ref, global_min[0]);
}