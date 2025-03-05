#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sedova_o_multiply_matrices_ccs/include/ops_seq.hpp"

namespace sedova_o_multiply_matrices_ccs_seq {
namespace {
std::vector<std::vector<double>> GenerateMatrix(int rows, int cols, int non_zero_count) {
  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> row_dist(0, rows - 1);
  std::uniform_int_distribution<> col_dist(0, cols - 1);
  std::uniform_real_distribution<> value_dist(-10.0, 10.0);

  int count = 0;
  while (count < non_zero_count) {
    int r = row_dist(gen);
    int c = col_dist(gen);

    if (matrix[r][c] == 0.0) {
      matrix[r][c] = value_dist(gen);
      ++count;
    }
  }

  return matrix;
}

std::vector<std::vector<double>> MultiplyMatrices(const std::vector<std::vector<double>> &a,
                                                  const std::vector<std::vector<double>> &b) {
  int rows_a = a.size();
  int cols_a = a[0].size();
  int cols_b = b[0].size();

  std::vector<std::vector<double>> result(rows_a, std::vector<double>(cols_b, 0.0));

  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      for (int k = 0; k < cols_a; ++k) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}
}  // namespace
}  // namespace sedova_o_multiply_matrices_ccs_seq

TEST(sedova_o_multiply_matrices_ccs_seq, test_pipeline_run) {
  int size = 256;
  int elements = 6553;
  std::vector<std::vector<double>> a;
  std::vector<std::vector<double>> b;

  a = sedova_o_multiply_matrices_ccs_seq::GenerateMatrix(size, size, elements);
  b = sedova_o_multiply_matrices_ccs_seq::GenerateMatrix(size, size, elements);

  std::vector<double> a_val;
  std::vector<int> a_row_ind;
  std::vector<int> a_col_ptr;
  int rows_a = a.size();
  int cols_a = a[0].size();

  std::vector<double> b_val;
  std::vector<int> b_row_ind;
  std::vector<int> b_col_ptr;
  int rows_b = b.size();
  int cols_b = b[0].size();

  std::vector<double> exp_c_val;
  std::vector<int> exp_c_row_ind;
  std::vector<int> exp_c_col_ptr;

  auto exp_c = sedova_o_multiply_matrices_ccs_seq::MultiplyMatrices(a, b);
  sedova_o_multiply_matrices_ccs_seq::Convertirovanie(exp_c, exp_c.size(), exp_c[0].size(), exp_c_val, exp_c_row_ind,
                                                      exp_c_col_ptr);

  std::vector<double> c_val;
  std::vector<int> c_row_ind;
  std::vector<int> c_col_ptr;

  sedova_o_multiply_matrices_ccs_seq::Convertirovanie(a, rows_a, cols_a, a_val, a_row_ind, a_col_ptr);
  sedova_o_multiply_matrices_ccs_seq::Convertirovanie(b, rows_b, cols_b, b_val, b_row_ind, b_col_ptr);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_a));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_a));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_b));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_b));
  task_data->inputs_count.emplace_back(1);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_val.data()));
  task_data->inputs_count.emplace_back(a_val.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_ind.data()));
  task_data->inputs_count.emplace_back(a_row_ind.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data->inputs_count.emplace_back(a_col_ptr.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_val.data()));
  task_data->inputs_count.emplace_back(b_val.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_ind.data()));
  task_data->inputs_count.emplace_back(b_row_ind.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data->inputs_count.emplace_back(b_col_ptr.size());
  
  c_val.resize(exp_c_val.size());
  c_row_ind.resize(exp_c_row_ind.size());
  c_col_ptr.resize(exp_c_col_ptr.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_val.data()));
  task_data->outputs_count.emplace_back(c_val.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_ind.data()));
  task_data->outputs_count.emplace_back(c_row_ind.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data->outputs_count.emplace_back(c_col_ptr.size());

  auto task = std::make_shared<sedova_o_multiply_matrices_ccs_seq::TestTaskSequential>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(exp_c_val, c_val);
  ASSERT_EQ(exp_c_row_ind, c_row_ind);
  ASSERT_EQ(exp_c_col_ptr, c_col_ptr);
}

TEST(sedova_o_multiply_matrices_ccs_seq, test_task_run) {
  int size = 256;
  int elements = 6553;
  std::vector<std::vector<double>> a;
  std::vector<std::vector<double>> b;

  a = sedova_o_multiply_matrices_ccs_seq::GenerateMatrix(size, size, elements);
  b = sedova_o_multiply_matrices_ccs_seq::GenerateMatrix(size, size, elements);

  std::vector<double> a_val;
  std::vector<int> a_row_ind;
  std::vector<int> a_col_ptr;
  int rows_a = a.size();
  int cols_a = a[0].size();

  std::vector<double> b_val;
  std::vector<int> b_row_ind;
  std::vector<int> b_col_ptr;
  int rows_b = b.size();
  int cols_b = b[0].size();

  std::vector<double> exp_c_val;
  std::vector<int> exp_c_row_ind;
  std::vector<int> exp_c_col_ptr;

  auto exp_c = sedova_o_multiply_matrices_ccs_seq::MultiplyMatrices(a, b);
  sedova_o_multiply_matrices_ccs_seq::Convertirovanie(exp_c, exp_c.size(), exp_c[0].size(), exp_c_val, exp_c_row_ind,
                                                      exp_c_col_ptr);

  std::vector<double> c_val;
  std::vector<int> c_row_ind;
  std::vector<int> c_col_ptr;

  sedova_o_multiply_matrices_ccs_seq::Convertirovanie(a, rows_a, cols_a, a_val, a_row_ind, a_col_ptr);
  sedova_o_multiply_matrices_ccs_seq::Convertirovanie(b, rows_b, cols_b, b_val, b_row_ind, b_col_ptr);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_a));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_a));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_b));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_b));
  task_data->inputs_count.emplace_back(1);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_val.data()));
  task_data->inputs_count.emplace_back(a_val.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_ind.data()));
  task_data->inputs_count.emplace_back(a_row_ind.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));
  task_data->inputs_count.emplace_back(a_col_ptr.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_val.data()));
  task_data->inputs_count.emplace_back(b_val.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_ind.data()));
  task_data->inputs_count.emplace_back(b_row_ind.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));
  task_data->inputs_count.emplace_back(b_col_ptr.size());

  c_val.resize(exp_c_val.size());
  c_row_ind.resize(exp_c_row_ind.size());
  c_col_ptr.resize(exp_c_col_ptr.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_val.data()));
  task_data->outputs_count.emplace_back(c_val.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_ind.data()));
  task_data->outputs_count.emplace_back(c_row_ind.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));
  task_data->outputs_count.emplace_back(c_col_ptr.size());

  auto task = std::make_shared<sedova_o_multiply_matrices_ccs_seq::TestTaskSequential>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(exp_c_val, c_val);
  ASSERT_EQ(exp_c_row_ind, c_row_ind);
  ASSERT_EQ(exp_c_col_ptr, c_col_ptr);
}
