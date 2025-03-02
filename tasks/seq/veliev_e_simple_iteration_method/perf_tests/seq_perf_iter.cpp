// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/veliev_e_simple_iteration_method/include/seq_header_iter.hpp"
namespace {
void GenerateStrictlyDiagonallyDominantMatrix(int size, std::vector<double> &matrix, std::vector<double> &rhs_vector) {
  matrix.resize(size * size);
  rhs_vector.resize(size);

  for (int row = 0; row < size; ++row) {
    double off_diag_sum = 0.0;
    matrix[(row * size) + row] = 2.0 * size;

    for (int col = 0; col < size; ++col) {
      if (row != col) {
        matrix[(row * size) + col] = 1.0;
        off_diag_sum += std::abs(matrix[(row * size) + col]);
      }
    }

    rhs_vector[row] = matrix[(row * size) + row] + off_diag_sum;
  }
}
}  // namespace

TEST(veliev_e_simple_iteration_method_seq, test_pipeline_run) {
  const int input_size = 5000;

  std::vector<double> matrix;
  std::vector<double> g;
  GenerateStrictlyDiagonallyDominantMatrix(input_size, matrix, g);

  std::vector<double> x(input_size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.push_back(input_size);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(g.data()));
  task_data_seq->inputs_count.push_back(input_size);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(input_size);

  auto test_task_sequential = std::make_shared<veliev_e_simple_iteration_method_seq::VelievSlaeIterSeq>(task_data_seq);

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

  std::vector<double> expected_solution(input_size, 1.0);

  double tolerance = 1e-6;
  for (int i = 0; i < input_size; ++i) {
    ASSERT_NEAR(x[i], expected_solution[i], tolerance);
  }
}

TEST(veliev_e_simple_iteration_method_seq, test_task_run) {
  const int input_size = 5000;

  std::vector<double> matrix;
  std::vector<double> g;
  GenerateStrictlyDiagonallyDominantMatrix(input_size, matrix, g);

  std::vector<double> x(input_size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.push_back(input_size);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(g.data()));
  task_data_seq->inputs_count.push_back(input_size);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(input_size);

  auto test_task_sequential = std::make_shared<veliev_e_simple_iteration_method_seq::VelievSlaeIterSeq>(task_data_seq);
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

  std::vector<double> expected_solution(input_size, 1.0);

  double tolerance = 1e-6;
  for (int i = 0; i < input_size; ++i) {
    ASSERT_NEAR(x[i], expected_solution[i], tolerance);
  }
}