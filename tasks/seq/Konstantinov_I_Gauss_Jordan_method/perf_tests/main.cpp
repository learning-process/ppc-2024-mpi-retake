#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/Konstantinov_I_Gauss_Jordan_method/include/ops_seq.hpp"

namespace konstantinov_i_gauss_jordan_method_seq {
namespace {
std::vector<double> GetRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(-20.0, 20.0);
  std::vector<double> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}
}  // namespace
}  // namespace konstantinov_i_gauss_jordan_method_seq

TEST(Konstantinov_i_gauss_jordan_method_seq, task_run) {
  int n = 500;
  std::vector<double> global_matrix = konstantinov_i_gauss_jordan_method_seq::GetRandomMatrix(n, n + 1);
  std::vector<double> global_result(n * (n + 1));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  task_data_seq->inputs_count.emplace_back(global_matrix.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  task_data_seq->outputs_count.emplace_back(global_result.size());

  auto task_sequential = std::make_shared<konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq>(task_data_seq);
  EXPECT_TRUE(task_sequential->ValidationImpl());
  task_sequential->PreProcessingImpl();
  task_sequential->RunImpl();
  task_sequential->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(Konstantinov_i_gauss_jordan_method_seq, pipeline_run) {
  int n = 500;
  std::vector<double> global_matrix = konstantinov_i_gauss_jordan_method_seq::GetRandomMatrix(n, n + 1);
  std::vector<double> global_result(n * (n + 1));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  task_data_seq->inputs_count.emplace_back(global_matrix.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  task_data_seq->outputs_count.emplace_back(global_result.size());

  auto task_sequential = std::make_shared<konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq>(task_data_seq);
  EXPECT_TRUE(task_sequential->ValidationImpl());
  task_sequential->PreProcessingImpl();
  task_sequential->RunImpl();
  task_sequential->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);
}