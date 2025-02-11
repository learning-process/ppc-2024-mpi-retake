#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/deryabin_m_cannons_algorithm/include/ops_seq.hpp"

TEST(deryabin_m_cannons_algorithm_seq, test_pipeline_run) {
  std::vector<double> input_matrix_a = std::vector<double>(10000, 0);
  std::vector<double> input_matrix_b = std::vector<double>(10000, 0);
  std::vector<double> output_matrix_c = std::vector<double>(10000, 0);
  for (unsigned short dimension = 0; dimension < 100; dimension++) {
    input_matrix_a[dimension * 101] = 1;
    input_matrix_b[dimension * 101] = 1;
  }
  std::vector<std::vector<double>> in_matrix_a(1, input_matrix_a);
  std::vector<std::vector<double>> in_matrix_b(1, input_matrix_b);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
  task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_seq->outputs_count.emplace_back(out_matrix_c.size());

  auto cannons_algorithm_TaskSequential =
      std::make_shared<deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential>(task_data_seq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(cannons_algorithm_TaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(in_matrix_a[0], out_matrix_c[0]);
}

TEST(deryabin_m_cannons_algorithm_seq, test_task_run) {
  std::vector<double> input_matrix_a = std::vector<double>(10000, 0);
  std::vector<double> input_matrix_b = std::vector<double>(10000, 0);
  std::vector<double> output_matrix_c = std::vector<double>(10000, 0);
  for (unsigned short dimension = 0; dimension < 100; dimension++) {
    input_matrix_a[dimension * 101] = 1;
    input_matrix_b[dimension * 101] = 1;
  }
  std::vector<std::vector<double>> in_matrix_a(1, input_matrix_a);
  std::vector<std::vector<double>> in_matrix_b(1, input_matrix_b);
  std::vector<std::vector<double>> out_matrix_c(1, output_matrix_c);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix_a.size());
  task_data_seq->inputs_count.emplace_back(input_matrix_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_c.data()));
  task_data_seq->outputs_count.emplace_back(out_matrix_c.size());

  auto cannons_algorithm_TaskSequential =
      std::make_shared<deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential>(task_data_seq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(cannons_algorithm_TaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(in_matrix_a[0], out_matrix_c[0]);
}
