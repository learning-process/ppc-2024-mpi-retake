#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_seq.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq {

std::vector<double> getRandomMatrix(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-1000, 1000);
  std::vector<double> matrix(sz);
  for (int i = 0; i < sz; ++i) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_pipeline_run) {
  constexpr int cols = 101;
  constexpr int rows = 100;

  std::vector<double> matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::getRandomMatrix(cols * rows);
  std::vector<double> res(cols - 1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(matrix.size());
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  auto gauss_seq =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<double>>(
          task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(gauss_seq);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_task_run) {
  constexpr int cols = 101;
  constexpr int rows = 100;

  std::vector<double> matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::getRandomMatrix(cols * rows);
  std::vector<double> res(cols - 1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(matrix.size());
  task_data_seq->inputs_count.emplace_back(cols);
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  auto gauss_seq =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential<double>>(
          task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(gauss_seq);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
