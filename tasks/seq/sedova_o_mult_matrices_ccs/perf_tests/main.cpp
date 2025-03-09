#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sedova_o_mult_matrices_ccs/include/ops_seq.hpp"

TEST(sedova_o_test_task_seq, test_pipeline_run) {
  std::vector<std::vector<double>> matrix_A = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<std::vector<double>> matrix_B = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> A;
  std::vector<int> row_ind_A;
  std::vector<int> col_ind_A;
  sedova_o_test_task_seq::ConvertToCCS(matrix_A, A, row_ind_A, col_ind_A);
  std::vector<double> B;
  std::vector<int> row_ind_B;
  std::vector<int> col_ind_B;
  sedova_o_test_task_seq::ConvertToCCS(matrix_B, B, row_ind_B, col_ind_B);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<std::vector<double>> out(matrix_A.size(), std::vector<double>(matrix_B[0].size(), 0));
  sedova_o_test_task_seq::FillData(task_data, matrix_A.size(), matrix_A[0].size(), matrix_B.size(), matrix_B[0].size(),
                                         A, row_ind_A, col_ind_A, B, row_ind_B, col_ind_B, out);
  // Create Task
  auto TestTaskSequential = std::make_shared<sedova_o_test_task_seq::TestTaskSequential>(task_data);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  std::vector<std::vector<double>> ans(matrix_A.size(), std::vector<double>(matrix_B[0].size(), 0));
  for (size_t i = 0; i < out.size(); ++i) {
    auto *ptr = reinterpret_cast<double *>(task_data->outputs[i]);
    ans[i] = std::vector(ptr, ptr + matrix_B.size());
  }
  std::vector<std::vector<double>> check_result = {{2, 0, 6}, {0, 14, 0}, {4, 0, 12}};
  ASSERT_EQ(check_result, ans);
}

TEST(sedova_o_test_task_seq, test_task_run) {
  std::vector<std::vector<double>> matrix_A = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<std::vector<double>> matrix_B = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> A;
  std::vector<int> row_ind_A;
  std::vector<int> col_ind_A;
  sedova_o_test_task_seq::ConvertToCCS(matrix_A, A, row_ind_A, col_ind_A);
  std::vector<double> B;
  std::vector<int> row_ind_B;
  std::vector<int> col_ind_B;
  sedova_o_test_task_seq::ConvertToCCS(matrix_B, B, row_ind_B, col_ind_B);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<std::vector<double>> out(matrix_A.size(), std::vector<double>(matrix_B[0].size(), 0));
  sedova_o_test_task_seq::FillData(task_data, matrix_A.size(), matrix_A[0].size(), matrix_B.size(), matrix_B[0].size(),
                                   A, row_ind_A, col_ind_A, B, row_ind_B, col_ind_B, out);
  // Create Task
  auto TestTaskSequential = std::make_shared<sedova_o_test_task_seq::TestTaskSequential>(task_data);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  std::vector<std::vector<double>> ans(matrix_A.size(), std::vector<double>(matrix_B[0].size(), 0));
  for (size_t i = 0; i < out.size(); ++i) {
    auto *ptr = reinterpret_cast<double *>(task_data->outputs[i]);
    ans[i] = std::vector(ptr, ptr + matrix_B.size());
  }
  std::vector<std::vector<double>> check_result = {{2, 0, 6}, {0, 14, 0}, {4, 0, 12}};
  ASSERT_EQ(check_result, ans);
}