#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/agafeev_s_strassen_alg/include/strassen_seq.hpp"

static std::vector<double> matrix_Multiply(const std::vector<T>& A, const std::vector<T>& B, int rowColSize) {
  std::vector<double> C(rowColSize * rowColSize, 0);

  for (unsigned int i = 0; i < rowColSize; ++i) {
    for (unsigned int j = 0; j < rowColSize; ++j) {
      for (unsigned int k = 0; k < rowColSize; ++k) {
        C[i * rowColSize + j] += A[i * rowColSize + k] * B[k * rowColSize + j];
      }
    }
  }

  return C;
}

static std::vector<double> create_RandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(1337);
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  std::vector<double> matrix(row_size * column_size);
  for (unsigned int i = 0; i < matrix.size(); i++) matrix[i] = dist(rand_gen);

  return matrix;
}

TEST(agafeev_s_strassen_alg_seq, test_pipeline_run) {
  const int n = 1024;
  const int m = 1024;

  // Credate Data
  std::vector<double> in_matrix1 = create_RandomMatrix<int>(n, m);
  std::vector<double> in_matrix2 = create_RandomMatrix<int>(n, m);
  std::vector<double> out(n * m,0);

  // Create task_data
  std::shared_ptr<ppc::core::task_data> task_data = std::make_shared<ppc::core::task_data>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(in_matrix1.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(in_matrix2.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto testTaskSequental = std::make_shared<agafeev_s_strassen_alg_seq::MultiplMatrixSequental>(task_data);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequental);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  auto temp = matrix_Multiply(in_matrix1, in_matrix2, n);

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_FLOAT_EQ(temp[i], out[i]);
  }
}

TEST(agafeev_s_strassen_alg_seq, test_task_run) {
  const int n = 1024;
  const int m = 1024;

  // Credate Data
  std::vector<double> in_matrix1 = create_RandomMatrix<int>(n, m);
  std::vector<double> in_matrix2 = create_RandomMatrix<int>(n, m);
  std::vector<double> out(n * m);

  // Create task_data
  std::shared_ptr<ppc::core::task_data> task_data = std::make_shared<ppc::core::task_data>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix.data()));
  task_data->inputs_count.emplace_back(in_matrix.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto testTaskSequental = std::make_shared<agafeev_s_strassen_alg_seq::MultiplMatrixSequental<int>>(task_data);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequental);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  auto temp = agafeev_s_strassen_alg_seq::matrix_Multiply(in_matrix1, in_matrix2, n);

  ASSERT_EQ(temp, out);
}
