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
std::vector<double> GenerateInvertibleMatrix(int size) {
  std::vector<double> matrix(size * (size + 1));
  std::random_device rd;
  std::mt19937 gen(rd());
  double lower_limit = -100.0;
  double upper_limit = 100.0;
  std::uniform_real_distribution<> dist(lower_limit, upper_limit);

  for (int i = 0; i < size; ++i) {
    double row_sum = 0.0;
    double diag = ((i * (size + 1)) + i);
    for (int j = 0; j < size + 1; ++j) {
      if (i != j) {
        matrix[(i * (size + 1)) + j] = dist(gen);
        row_sum += std::abs(matrix[(i * (size + 1)) + j]);
      }
    }
    std::size_t diag_index = static_cast<std::size_t>(std::round(diag));
    matrix[diag_index] = static_cast<double>(row_sum + 1);
  }

  return matrix;
}
}  // namespace
}  // namespace konstantinov_i_gauss_jordan_method_seq

TEST(Konstantinov_i_gauss_jordan_method_seq, test_pipeline_run) {
  int size = 500;

  std::vector<double> matrix = konstantinov_i_gauss_jordan_method_seq::GenerateInvertibleMatrix(size);

  std::vector<double> output_data(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(matrix.size() / (size + 1));
  task_data_seq->inputs_count.emplace_back(matrix.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_seq->outputs_count.emplace_back(output_data.size());

  auto gauss_task_sequential =
      std::make_shared<konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq>(task_data_seq);

  ASSERT_EQ(gauss_task_sequential->ValidationImpl(), true);
  gauss_task_sequential->PreProcessingImpl();
  gauss_task_sequential->RunImpl();
  gauss_task_sequential->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(gauss_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(output_data.size(), static_cast<std::size_t>(size));
}

TEST(Konstantinov_i_gauss_jordan_method_seq, test_task_run) {
  int size = 500;

  auto matrix = konstantinov_i_gauss_jordan_method_seq::GenerateInvertibleMatrix(size);

  std::vector<double> output_data(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(matrix.size() / (size + 1));
  task_data_seq->inputs_count.emplace_back(matrix.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_seq->outputs_count.emplace_back(output_data.size());

  auto gauss_task_sequential =
      std::make_shared<konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq>(task_data_seq);
  ASSERT_EQ(gauss_task_sequential->ValidationImpl(), true);
  gauss_task_sequential->PreProcessingImpl();
  gauss_task_sequential->RunImpl();
  gauss_task_sequential->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(gauss_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(output_data.size(), static_cast<std::size_t>(size));
}