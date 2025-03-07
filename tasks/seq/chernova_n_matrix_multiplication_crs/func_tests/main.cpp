#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/chernova_n_matrix_multiplication_crs/include/ops_seq.hpp"

#include <gtest/gtest.h>
#include <vector>



namespace {
chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS generateRandomCRS(int size,
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
      if (static_cast<double>(gen()) / gen.max() < density) {
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
chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS generateIdentityCRS(int n) {
  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrix;
  if (n <= 0) return matrix;

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
bool compareCRS(const chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS& a,
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
  std::vector<double> valuesA = {1.0, 2.0, 3.0};
  std::vector<int> colIndicesA = {0, 1, 2};
  std::vector<int> rowPtrA = {0, 1, 2, 3};

  std::vector<double> valuesB = {1.0, 1.0, 1.0};
  std::vector<int> colIndicesB = {0, 1, 2};
  std::vector<int> rowPtrB = {0, 1, 2, 3};

  std::vector<double> expectedValues = {1.0, 2.0, 3.0};
  std::vector<int> expectedColIndices = {0, 1, 2};
  std::vector<int> expectedRowPtr = {0, 1, 2, 3};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  SetupTaskData(valuesA, colIndicesA, rowPtrA, task_data_seq);
  SetupTaskData(valuesB, colIndicesB, rowPtrB, task_data_seq);

  std::vector<double> resultValues(expectedValues.size());
  std::vector<int> resultColIndices(expectedColIndices.size());
  std::vector<int> resultRowPtr(expectedRowPtr.size());

    SetupOutData(resultValues, resultColIndices, resultRowPtr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(resultValues, expectedValues);
  EXPECT_EQ(resultColIndices, expectedColIndices);
  EXPECT_EQ(resultRowPtr, expectedRowPtr);
} 
TEST(chernova_n_matrix_multiplication_crs_seq, test_mul_5x5_seq) {
  std::vector<double> valuesA = {3.0, 2.0, 5.0, 1.0, 4.0};
  std::vector<int> colIndicesA = {2, 1, 4, 0, 3};
  std::vector<int> rowPtrA = {0, 1, 2, 3, 4, 5};

  std::vector<double> valuesB = {2.0, 3.0, 5.0, 1.0, 4.0};
  std::vector<int> colIndicesB = {1, 3, 0, 4, 2};
  std::vector<int> rowPtrB = {0, 1, 2, 3, 4, 5};

  std::vector<double> expectedValues = {15.0, 6.0, 20.0, 2.0, 4.0};
  std::vector<int> expectedColIndices = {0, 3, 2, 1, 4};
  std::vector<int> expectedRowPtr = {0, 1, 2, 3, 4, 5};

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  SetupTaskData(valuesA, colIndicesA, rowPtrA, task_data_seq);
  SetupTaskData(valuesB, colIndicesB, rowPtrB, task_data_seq);

  std::vector<double> resultValues(expectedValues.size());
  std::vector<int> resultColIndices(expectedColIndices.size());
  std::vector<int> resultRowPtr(expectedRowPtr.size());

  SetupOutData(resultValues, resultColIndices, resultRowPtr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(resultValues, expectedValues);
  EXPECT_EQ(resultColIndices, expectedColIndices);
  EXPECT_EQ(resultRowPtr, expectedRowPtr);
}

TEST(chernova_n_matrix_multiplication_crs_seq, test_mul_10x10_seq) {
  std::vector<double> valuesA = {3.0, 7.0, 2.0, 5.0, 6.0, 4.0, 9.0, 1.0, 8.0, 10.0};
  std::vector<int> colIndicesA = {2, 5, 9, 1, 4, 7, 0, 3, 6, 8};
  std::vector<int> rowPtrA = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<double> valuesB = {2.0, 5.0, 1.0, 3.0, 6.0, 8.0, 4.0, 7.0, 9.0, 2.0};
  std::vector<int> colIndicesB = {4, 7, 3, 9, 0, 2, 5, 1, 6, 8};
  std::vector<int> rowPtrB = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<double> expectedValues = {3.0, 56.0, 4.0, 25.0, 36.0, 28.0, 18.0, 3.0, 32.0, 90.0};
  std::vector<int> expectedColIndices = {3, 2, 8, 7, 0, 1, 4, 9, 5, 6};
  std::vector<int> expectedRowPtr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  SetupTaskData(valuesA, colIndicesA, rowPtrA, task_data_seq);
  SetupTaskData(valuesB, colIndicesB, rowPtrB, task_data_seq);

  std::vector<double> resultValues(expectedValues.size());
  std::vector<int> resultColIndices(expectedColIndices.size());
  std::vector<int> resultRowPtr(expectedRowPtr.size());

  SetupOutData(resultValues, resultColIndices, resultRowPtr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(resultValues, expectedValues);
  EXPECT_EQ(resultColIndices, expectedColIndices);
  EXPECT_EQ(resultRowPtr, expectedRowPtr);
}

TEST(chernova_n_matrix_multiplication_crs_seq, test_mul_15x15_seq) {
  std::vector<double> valuesA = {3.0, 7.0, 2.0, 5.0, 6.0, 4.0, 9.0, 1.0, 8.0, 10.0, 5.0, 2.0, 4.0, 7.0, 6.0};
  std::vector<int> colIndicesA = {2, 5, 9, 1, 4, 7, 0, 3, 6, 8, 10, 12, 13, 14, 11};
  std::vector<int> rowPtrA = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  std::vector<double> valuesB = {2.0, 5.0, 1.0, 3.0, 6.0, 8.0, 4.0, 7.0, 9.0, 2.0, 5.0, 3.0, 1.0, 4.0, 7.0};
  std::vector<int> colIndicesB = {4, 7, 3, 9, 0, 2, 5, 1, 6, 8, 10, 12, 13, 14, 11};
  std::vector<int> rowPtrB = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  std::vector<double> expectedValues = {3.0,  56.0, 4.0,  25.0, 36.0, 28.0, 18.0, 3.0,
                                        32.0, 90.0, 25.0, 2.0,  16.0, 49.0, 18.0};
  std::vector<int> expectedColIndices = {3, 2, 8, 7, 0, 1, 4, 9, 5, 6, 10, 13, 14, 11, 12};
  std::vector<int> expectedRowPtr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  SetupTaskData(valuesA, colIndicesA, rowPtrA, task_data_seq);
  SetupTaskData(valuesB, colIndicesB, rowPtrB, task_data_seq);

  std::vector<double> resultValues(expectedValues.size());
  std::vector<int> resultColIndices(expectedColIndices.size());
  std::vector<int> resultRowPtr(expectedRowPtr.size());

  SetupOutData(resultValues, resultColIndices, resultRowPtr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(resultValues, expectedValues);
  EXPECT_EQ(resultColIndices, expectedColIndices);
  EXPECT_EQ(resultRowPtr, expectedRowPtr);
}

TEST(chernova_n_matrix_multiplication_crs_seq, random_matrix_10) {
  const int matrix_size = 10;
  const double density = 0.1;

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrixA =
      generateRandomCRS(matrix_size, density);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS matrixB =
      generateIdentityCRS(matrix_size);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  SetupTaskData(matrixA.values, matrixA.col_indices, matrixA.row_ptr, task_data_seq);
  SetupTaskData(matrixB.values, matrixB.col_indices, matrixB.row_ptr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential::SparseMatrixCRS result;
  result.values.resize(matrixA.values.size());
  result.col_indices.resize(matrixA.col_indices.size());
  result.row_ptr.resize(matrixA.row_ptr.size());

  SetupOutData(result.values, result.col_indices, result.row_ptr, task_data_seq);

  chernova_n_matrix_multiplication_crs_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_TRUE(compareCRS(matrixA, result));
}
 
