#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/chernova_n_matrix_multiplication_crs/include/ops_seq.hpp"

namespace {
chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS GenerateRandomCrs(int size,
                                                                                                double k_density,
                                                                                                int k_seed) {
  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrix;
  std::mt19937 gen(k_seed);
  std::uniform_real_distribution<double> val_dist(1.0, 10.0);

  matrix.row_ptr.push_back(0);
  int total_non_zero = 0;

  for (int i = 0; i < size; ++i) {
    int nnz_in_row = 0;
    for (int j = 0; j < size; ++j) {
      if (static_cast<double>(gen()) / std::mt19937::max() < k_density) {
        matrix.values.push_back(val_dist(gen));
        matrix.col_indices.push_back(j);
        nnz_in_row++;
      }
    }
    total_non_zero += nnz_in_row;
    matrix.row_ptr.push_back(total_non_zero);
  }
  return matrix;
}

chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS GenerateIdentityCrs(int n) {
  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrix;
  matrix.values.resize(n, 1.0);
  matrix.col_indices.resize(n);
  matrix.row_ptr.resize(n + 1);

  for (int i = 0; i < n; ++i) {
    matrix.col_indices[i] = i;
    matrix.row_ptr[i] = i;
  }
  matrix.row_ptr[n] = n;
  return matrix;
}
bool CompareCrs(const chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS& a,
                const chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS& b) {
  return a.values == b.values && a.col_indices == b.col_indices && a.row_ptr == b.row_ptr;
}
void SetupTaskData(std::vector<double>& values, std::vector<int>& columns, std::vector<int>& rows,
                   std::shared_ptr<ppc::core::TaskData>& task_data) {
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(values.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(columns.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(rows.data()));
  task_data->inputs_count.emplace_back(values.size());
  task_data->inputs_count.emplace_back(columns.size());
  task_data->inputs_count.emplace_back(rows.size());
}
}  // namespace

TEST(chernova_n_matrix_multiplication_crs_seq, test_pipeline_run) {
  constexpr int kSize = 2000;
  constexpr double kDensity = 0.1;
  constexpr int kSeed = 42;

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrix_a =
      GenerateRandomCrs(kSize, kDensity, kSeed);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrix_b = GenerateIdentityCrs(kSize);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  SetupTaskData(matrix_a.values, matrix_a.col_indices, matrix_a.row_ptr, task_data_seq);
  SetupTaskData(matrix_b.values, matrix_b.col_indices, matrix_b.row_ptr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS result;
  result.values = std::vector<double>(matrix_a.values.size());
  result.col_indices = std::vector<int>(matrix_a.col_indices.size());
  result.row_ptr = std::vector<int>(matrix_a.row_ptr.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.values.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.col_indices.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.row_ptr.data()));
  task_data_seq->outputs_count.emplace_back(result.values.size());

  auto test_task = std::make_shared<chernova_n_matrix_multiplication_crs_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_TRUE(CompareCrs(matrix_a, result));
}

TEST(chernova_n_matrix_multiplication_crs_seq, test_task_run) {
  constexpr int kSize = 2000;
  constexpr double kDensity = 0.1;
  constexpr int kSeed = 42;

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrix_a =
      GenerateRandomCrs(kSize, kDensity, kSeed);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrix_b = GenerateIdentityCrs(kSize);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  SetupTaskData(matrix_a.values, matrix_a.col_indices, matrix_a.row_ptr, task_data_seq);
  SetupTaskData(matrix_b.values, matrix_b.col_indices, matrix_b.row_ptr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS result;
  result.values = std::vector<double>(matrix_a.values.size());
  result.col_indices = std::vector<int>(matrix_a.col_indices.size());
  result.row_ptr = std::vector<int>(matrix_a.row_ptr.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.values.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.col_indices.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.row_ptr.data()));
  task_data_seq->outputs_count.emplace_back(result.values.size());

  auto test_task = std::make_shared<chernova_n_matrix_multiplication_crs_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  EXPECT_TRUE(CompareCrs(matrix_a, result));
}