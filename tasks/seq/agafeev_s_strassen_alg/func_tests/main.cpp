#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/agafeev_s_strassen_alg/include/strassen_seq.hpp"

static std::vector<double> MatrixMultiply(const std::vector<double>& a, const std::vector<double>& b,
                                          int row_col_size) {
  std::vector<double> c(row_col_size * row_col_size, 0);

  for (int i = 0; i < row_col_size; ++i) {
    for (int j = 0; j < row_col_size; ++j) {
      for (int k = 0; k < row_col_size; ++k) {
        c[(i * row_col_size) + j] += a[(i * row_col_size) + k] * b[(k * row_col_size) + j];
      }
    }
  }

  return c;
}

static std::vector<double> CreateRandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(time(nullptr));
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  std::vector<double> matrix(row_size * column_size);
  for (unsigned int i = 0; i < matrix.size(); i++) {
    matrix[i] = dist(rand_gen);
  }

  return matrix;
}

TEST(agafeev_s_strassen_alg_seq, matmul_1x1) {
  const int rows = 1;
  const int columns = 1;

  // Create data
  std::vector<double> in_matrix1 = CreateRandomMatrix(rows, columns);
  std::vector<double> in_matrix2 = CreateRandomMatrix(rows, columns);
  std::vector<double> out(rows * columns, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(columns);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(columns);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_strassen_alg_seq::MultiplMatrixSequental test_task(task_data);
  bool is_valid = test_task.ValidationImpl();
  ASSERT_EQ(is_valid, true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();
  auto right_answer = MatrixMultiply(in_matrix1, in_matrix2, rows);

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_FLOAT_EQ(right_answer[i], out[i]);
  }
}

TEST(agafeev_s_strassen_alg_seq, matmul_2x2) {
  const int rows = 2;
  const int columns = 2;

  // Create data
  std::vector<double> in_matrix1 = CreateRandomMatrix(rows, columns);
  std::vector<double> in_matrix2 = CreateRandomMatrix(rows, columns);
  std::vector<double> out(rows * columns, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(columns);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(columns);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_strassen_alg_seq::MultiplMatrixSequental test_task(task_data);
  bool is_valid = test_task.ValidationImpl();
  ASSERT_EQ(is_valid, true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();
  auto right_answer = MatrixMultiply(in_matrix1, in_matrix2, rows);

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_FLOAT_EQ(right_answer[i], out[i]);
  }
}

TEST(agafeev_s_strassen_alg_seq, matmul_64x64) {
  const int rows = 2;
  const int columns = 2;

  // Create data
  std::vector<double> in_matrix1 = CreateRandomMatrix(rows, columns);
  std::vector<double> in_matrix2 = CreateRandomMatrix(rows, columns);
  std::vector<double> out(rows * columns, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(columns);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(columns);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_strassen_alg_seq::MultiplMatrixSequental test_task(task_data);
  bool is_valid = test_task.ValidationImpl();
  ASSERT_EQ(is_valid, true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();
  auto right_answer = MatrixMultiply(in_matrix1, in_matrix2, rows);

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_FLOAT_EQ(right_answer[i], out[i]);
  }
}

TEST(agafeev_s_strassen_alg_seq, wrong_rowcolumn_valid) {
  // Create data
  std::vector<double> in_matrix1 = CreateRandomMatrix(3, 5);
  std::vector<double> in_matrix2 = CreateRandomMatrix(5, 4);
  std::vector<double> out(3 * 4, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(5);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(4);
  task_data->inputs_count.emplace_back(4);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_strassen_alg_seq::MultiplMatrixSequental test_task(task_data);
  bool is_valid = test_task.ValidationImpl();
  ASSERT_EQ(is_valid, false);
}