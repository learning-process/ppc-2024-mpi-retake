#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/Konstantinov_I_sum_of_vector_elements/include/ops_seq.hpp"

std::vector<int> Konstantinov_I_sum_of_vector_elements_seq::generate_rand_vector(int size, int lower_bound,
                                                                                 int upper_bound) {
  std::vector<int> result(size);
  for (int i = 0; i < size; i++) {
    result[i] = lower_bound + rand() % (upper_bound - lower_bound + 1);
  }
  return result;
}

std::vector<std::vector<int>> Konstantinov_I_sum_of_vector_elements_seq::generate_rand_matrix(int rows, int columns,
                                                                                              int lower_bound,
                                                                                              int upper_bound) {
  std::vector<std::vector<int>> result(rows);
  for (int i = 0; i < rows; i++) {
    result[i] = Konstantinov_I_sum_of_vector_elements_seq::generate_rand_vector(columns, lower_bound, upper_bound);
  }
  return result;
  return std::vector<std::vector<int>>();
}

TEST(Konstantinov_I_sum_of_vector_seq, test_pipeline_run) {
  int rows = 10000;
  int columns = 10000;
  int result;
  std::vector<std::vector<int>> input = Konstantinov_I_sum_of_vector_elements_seq::generate_rand_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  // Create Task
  auto test = std::make_shared<Konstantinov_I_sum_of_vector_elements_seq::SumVecElemSequential>(taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
  ASSERT_EQ(sum, result);
}

TEST(Konstantinov_I_sum_of_vector_seq, test_task_run) {
  int rows = 10000;
  int columns = 10000;
  int result;
  std::vector<std::vector<int>> input = Konstantinov_I_sum_of_vector_elements_seq::generate_rand_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  // Create Task
  auto test = std::make_shared<Konstantinov_I_sum_of_vector_elements_seq::SumVecElemSequential>(taskDataPar);
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->TaskRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
  ASSERT_EQ(sum, result);
}