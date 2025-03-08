#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/chernova_n_matrix_multiplication_crs/include/ops_seq.hpp"

namespace {
chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS GenerateRandomCrs(int size,
                                                                                                double density,
                                                                                                int seed = 42) {
  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrix;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> value_dist(1.0, 10.0);
  std::uniform_int_distribution<int> col_dist(0, size - 1);

  matrix.row_ptr.push_back(0);
  int total_non_zero = 0;

  for (int i = 0; i < size; ++i) {
    int non_zero_in_row = 0;
    for (int j = 0; j < size; ++j) {
      if (static_cast<double>(gen()) / std::mt19937::max() < density) {
        matrix.values.push_back(value_dist(gen));
        matrix.col_indices.push_back(j);
        non_zero_in_row++;
      }
    }
    total_non_zero += non_zero_in_row;
    matrix.row_ptr.push_back(total_non_zero);
  }
  return matrix;
}
chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS GenerateIdentityCrs(int n) {
  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrix;
  if (n <= 0) {
    return matrix;
  }

  matrix.values.resize(n, 1.0);
  matrix.col_indices.resize(n);
  matrix.row_ptr.resize(n + 1);

  for (int i = 0; i < n; ++i) {
    matrix.col_indices[i] = i;
  }

  for (int i = 0; i <= n; ++i) {
    matrix.row_ptr[i] = i;
  }
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
void SetupOutData(std::vector<double>& values, std::vector<int>& columns, std::vector<int>& rows,
                  std::shared_ptr<ppc::core::TaskData>& task_data) {
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(values.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(columns.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(rows.data()));
  task_data->outputs_count.emplace_back(values.size());
}
}  // namespace

TEST(chernova_n_matrix_multiplication_crs_seq, test_mul_3x3_seq) {
  std::vector<double> values_a = {1.0, 2.0, 3.0};
  std::vector<int> col_indices_a = {0, 1, 2};
  std::vector<int> row_ptr_a = {0, 1, 2, 3};

  std::vector<double> values_b = {1.0, 1.0, 1.0};
  std::vector<int> col_indices_b = {0, 1, 2};
  std::vector<int> row_ptr_b = {0, 1, 2, 3};

  std::vector<double> expected_values = {1.0, 2.0, 3.0};
  std::vector<int> expected_col_indices = {0, 1, 2};
  std::vector<int> expected_row_ptr = {0, 1, 2, 3};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  SetupTaskData(values_a, col_indices_a, row_ptr_a, task_data_seq);
  SetupTaskData(values_b, col_indices_b, row_ptr_b, task_data_seq);

  std::vector<double> result_values(expected_values.size());
  std::vector<int> result_col_indices(expected_col_indices.size());
  std::vector<int> result_row_ptr(expected_row_ptr.size());

  SetupOutData(result_values, result_col_indices, result_row_ptr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(result_values, expected_values);
  EXPECT_EQ(result_col_indices, expected_col_indices);
  EXPECT_EQ(result_row_ptr, expected_row_ptr);
}
TEST(chernova_n_matrix_multiplication_crs_seq, test_mul_5x5_seq) {
  std::vector<double> values_a = {3.0, 2.0, 5.0, 1.0, 4.0};
  std::vector<int> col_indices_a = {2, 1, 4, 0, 3};
  std::vector<int> row_ptr_a = {0, 1, 2, 3, 4, 5};

  std::vector<double> values_b = {2.0, 3.0, 5.0, 1.0, 4.0};
  std::vector<int> col_indices_b = {1, 3, 0, 4, 2};
  std::vector<int> row_ptr_b = {0, 1, 2, 3, 4, 5};

  std::vector<double> expected_values = {15.0, 6.0, 20.0, 2.0, 4.0};
  std::vector<int> expected_col_indices = {0, 3, 2, 1, 4};
  std::vector<int> expected_row_ptr = {0, 1, 2, 3, 4, 5};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  SetupTaskData(values_a, col_indices_a, row_ptr_a, task_data_seq);
  SetupTaskData(values_b, col_indices_b, row_ptr_b, task_data_seq);

  std::vector<double> result_values(expected_values.size());
  std::vector<int> result_col_indices(expected_col_indices.size());
  std::vector<int> result_row_ptr(expected_row_ptr.size());

  SetupOutData(result_values, result_col_indices, result_row_ptr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(result_values, expected_values);
  EXPECT_EQ(result_col_indices, expected_col_indices);
  EXPECT_EQ(result_row_ptr, expected_row_ptr);
}

TEST(chernova_n_matrix_multiplication_crs_seq, test_mul_10x10_seq) {
  std::vector<double> values_a = {3.0, 7.0, 2.0, 5.0, 6.0, 4.0, 9.0, 1.0, 8.0, 10.0};
  std::vector<int> col_indices_a = {2, 5, 9, 1, 4, 7, 0, 3, 6, 8};
  std::vector<int> row_ptr_a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<double> values_b = {2.0, 5.0, 1.0, 3.0, 6.0, 8.0, 4.0, 7.0, 9.0, 2.0};
  std::vector<int> col_indices_b = {4, 7, 3, 9, 0, 2, 5, 1, 6, 8};
  std::vector<int> row_ptr_b = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<double> expected_values = {3.0, 56.0, 4.0, 25.0, 36.0, 28.0, 18.0, 3.0, 32.0, 90.0};
  std::vector<int> expected_col_indices = {3, 2, 8, 7, 0, 1, 4, 9, 5, 6};
  std::vector<int> expected_row_ptr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  SetupTaskData(values_a, col_indices_a, row_ptr_a, task_data_seq);
  SetupTaskData(values_b, col_indices_b, row_ptr_b, task_data_seq);

  std::vector<double> result_values(expected_values.size());
  std::vector<int> result_col_indices(expected_col_indices.size());
  std::vector<int> result_row_ptr(expected_row_ptr.size());

  SetupOutData(result_values, result_col_indices, result_row_ptr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(result_values, expected_values);
  EXPECT_EQ(result_col_indices, expected_col_indices);
  EXPECT_EQ(result_row_ptr, expected_row_ptr);
}

TEST(chernova_n_matrix_multiplication_crs_seq, test_mul_15x15_seq) {
  std::vector<double> values_a = {3.0, 7.0, 2.0, 5.0, 6.0, 4.0, 9.0, 1.0, 8.0, 10.0, 5.0, 2.0, 4.0, 7.0, 6.0};
  std::vector<int> col_indices_a = {2, 5, 9, 1, 4, 7, 0, 3, 6, 8, 10, 12, 13, 14, 11};
  std::vector<int> row_ptr_a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  std::vector<double> values_b = {2.0, 5.0, 1.0, 3.0, 6.0, 8.0, 4.0, 7.0, 9.0, 2.0, 5.0, 3.0, 1.0, 4.0, 7.0};
  std::vector<int> col_indices_b = {4, 7, 3, 9, 0, 2, 5, 1, 6, 8, 10, 12, 13, 14, 11};
  std::vector<int> row_ptr_b = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  std::vector<double> expected_values = {3.0,  56.0, 4.0,  25.0, 36.0, 28.0, 18.0, 3.0,
                                         32.0, 90.0, 25.0, 2.0,  16.0, 49.0, 18.0};
  std::vector<int> expected_col_indices = {3, 2, 8, 7, 0, 1, 4, 9, 5, 6, 10, 13, 14, 11, 12};
  std::vector<int> expected_row_ptr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  SetupTaskData(values_a, col_indices_a, row_ptr_a, task_data_seq);
  SetupTaskData(values_b, col_indices_b, row_ptr_b, task_data_seq);

  std::vector<double> result_values(expected_values.size());
  std::vector<int> result_col_indices(expected_col_indices.size());
  std::vector<int> result_row_ptr(expected_row_ptr.size());

  SetupOutData(result_values, result_col_indices, result_row_ptr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(result_values, expected_values);
  EXPECT_EQ(result_col_indices, expected_col_indices);
  EXPECT_EQ(result_row_ptr, expected_row_ptr);
}

TEST(chernova_n_matrix_multiplication_crs_seq, random_matrix_10) {
  const int matrix_size = 10;
  const double density = 0.1;

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrix_a =
      GenerateRandomCrs(matrix_size, density);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrix_b =
      GenerateIdentityCrs(matrix_size);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  SetupTaskData(matrix_a.values, matrix_a.col_indices, matrix_a.row_ptr, task_data_seq);
  SetupTaskData(matrix_b.values, matrix_b.col_indices, matrix_b.row_ptr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS result;
  result.values.resize(matrix_a.values.size());
  result.col_indices.resize(matrix_a.col_indices.size());
  result.row_ptr.resize(matrix_a.row_ptr.size());

  SetupOutData(result.values, result.col_indices, result.row_ptr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_TRUE(CompareCrs(matrix_a, result));
}
