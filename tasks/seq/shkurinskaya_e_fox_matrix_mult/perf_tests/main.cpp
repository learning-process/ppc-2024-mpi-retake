#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/shkurinskaya_e_fox_matrix_mult/include/ops_sec.hpp"

TEST(shkurinskaya_e_fox_mat_mul_seq, test_pipline_run) {
  int matrix_size = 576;
  std::vector<double> in1 = shkurinskaya_e_fox_mat_mul_seq::GetRandomMatrix(matrix_size, matrix_size);
  std::vector<double> in2 = shkurinskaya_e_fox_mat_mul_seq::GetRandomMatrix(matrix_size, matrix_size);
  std::vector<double> out(matrix_size * matrix_size), ans(matrix_size * matrix_size, 0.0);

  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      for (int k = 0; k < matrix_size; ++k) {
        ans[(i * matrix_size) + j] += in1[(i * matrix_size) + k] * in2[(k * matrix_size) + j];
      }
    }
  }

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(matrix_size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(matrix_size);

  // crate task
  auto test_task_seq = std::make_shared<shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential>(task_data_seq);

  // create perf attrib
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_seq);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (int it = 0; it < (int)ans.size(); ++it) {
    ASSERT_NEAR(ans[it], out[it], 1);
  }
}

TEST(shkurinskaya_e_fox_mat_mul_seq, test_task_run) {
  int matrix_size = 576;
  std::vector<double> in1 = shkurinskaya_e_fox_mat_mul_seq::GetRandomMatrix(matrix_size, matrix_size);
  std::vector<double> in2 = shkurinskaya_e_fox_mat_mul_seq::GetRandomMatrix(matrix_size, matrix_size);
  std::vector<double> out(matrix_size * matrix_size), ans(matrix_size * matrix_size, 0.0);

  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      for (int k = 0; k < matrix_size; ++k) {
        ans[(i * matrix_size) + j] += in1[(i * matrix_size) + k] * in2[(k * matrix_size) + j];
      }
    }
  }
  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(matrix_size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(matrix_size);

  // crate task
  auto test_task_seq = std::make_shared<shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential>(task_data_seq);

  // create perf attrib
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_seq);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (int it = 0; it < (int)ans.size(); ++it) {
    ASSERT_NEAR(ans[it], out[it], 1);
  }
}
