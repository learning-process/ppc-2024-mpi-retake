#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/veliev_e_sum_values_by_rows_matrix/include/seq_rows_m_header.hpp"

TEST(veliev_e_sum_values_by_rows_matrix_seq, test_pipeline_run) {
  std::vector base_input = {100000000, 10000, 10000};

  // Create data
  std::vector<int> arr(base_input[0]);
  veliev_e_sum_values_by_rows_matrix_seq::GetRndMatrix(arr);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(base_input[1]);

  // Create Task
  auto test_task_sequential =
      std::make_shared<veliev_e_sum_values_by_rows_matrix_seq::SumValuesByRowsMatrixSeq>(task_data);

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
  std::vector<int> ref_for_check;
  veliev_e_sum_values_by_rows_matrix_seq::SeqProcForChecking(arr, base_input[2], ref_for_check);
  ASSERT_EQ(out, ref_for_check);
}

TEST(veliev_e_sum_values_by_rows_matrix_seq, test_task_run) {
  std::vector base_input = {100000000, 10000, 10000};

  // Create data
  std::vector<int> arr(base_input[0]);
  veliev_e_sum_values_by_rows_matrix_seq::GetRndMatrix(arr);
  std::vector<int> out(base_input[1]);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(base_input.data()));  // num_elem + rows + cols
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(arr.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(base_input[1]);

  // Create Task
  auto test_task_sequential =
      std::make_shared<veliev_e_sum_values_by_rows_matrix_seq::SumValuesByRowsMatrixSeq>(task_data);

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
  std::vector<int> ref_for_check;
  veliev_e_sum_values_by_rows_matrix_seq::SeqProcForChecking(arr, base_input[2], ref_for_check);
  ASSERT_EQ(out, ref_for_check);
}
