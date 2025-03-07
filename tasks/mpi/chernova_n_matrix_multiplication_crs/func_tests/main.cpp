#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <random>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/chernova_n_matrix_multiplication_crs/include/ops_mpi.hpp"


namespace {
chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS generateRandomCRS(int size,
                                                                                                double density,
                                                                                                int seed = 42) {
  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS matrix;
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
chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS generateIdentityCRS(int n) {
  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS matrix;
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

bool compareCRS(const chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS& a,
                const chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS& b) {
  return a.values == b.values && a.col_indices == b.col_indices && a.row_ptr == b.row_ptr;
}
}  // namespace


void SetupTaskData(std::vector<double>& values, std::vector<int>& columns, std::vector<int>& rows,
                   std::shared_ptr<ppc::core::TaskData>& task_data, const boost::mpi::communicator& world) {
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(values.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(columns.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(rows.data()));
    task_data->inputs_count.emplace_back(values.size());
    task_data->inputs_count.emplace_back(columns.size());
    task_data->inputs_count.emplace_back(rows.size());
  }
}
void SetupOutData(std::vector<double>& values, std::vector<int>& columns, std::vector<int>& rows,
                   std::shared_ptr<ppc::core::TaskData>& task_data, const boost::mpi::communicator& world) {
  if (world.rank() == 0) {
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(values.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(columns.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(rows.data()));
    task_data->outputs_count.emplace_back(values.size());
  }
}

TEST(chernova_n_matrix_multiplication_crs_mpi, test_sparse_10x10_parallel) {
  const int root_rank = 0;
  boost::mpi::communicator world_;

  std::vector<double> valuesA, valuesB;
  std::vector<int> colIndicesA, colIndicesB;
  std::vector<int> rowPtrA, rowPtrB;

  if (world_.rank() == root_rank) {
    valuesA = {3.0, 7.0, 2.0, 5.0, 6.0, 4.0, 9.0, 1.0, 8.0, 10.0};
    colIndicesA = {2, 5, 9, 1, 4, 7, 0, 3, 6, 8};
    rowPtrA = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    valuesB = {2.0, 5.0, 1.0, 3.0, 6.0, 8.0, 4.0, 7.0, 9.0, 2.0};
    colIndicesB = {4, 7, 3, 9, 0, 2, 5, 1, 6, 8};
    rowPtrB = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  }

  std::vector<double> expectedValues = {3.0, 56.0, 4.0, 25.0, 36.0, 28.0, 18.0, 3.0, 32.0, 90.0};
  std::vector<int> expectedColIndices = {3, 2, 8, 7, 0, 1, 4, 9, 5, 6};
  std::vector<int> expectedRowPtr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world_.rank() == root_rank) {

    SetupTaskData(valuesA, colIndicesA, rowPtrA, task_data, world_);
    SetupTaskData(valuesB, colIndicesB, rowPtrB, task_data, world_);

    std::vector<double> resultValues(expectedValues.size());
    std::vector<int> resultColIndices(expectedColIndices.size());
    std::vector<int> resultRowPtr(expectedRowPtr.size());

    SetupOutData(resultValues, resultColIndices, resultRowPtr, task_data, world_);
  }

  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI test_task(task_data);
  
  if (world_.rank() == root_rank) {
    ASSERT_TRUE(test_task.ValidationImpl());
  }
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world_.rank() == root_rank) {
    auto* output_values = reinterpret_cast<double*>(task_data->outputs[0]);
    auto* output_cols = reinterpret_cast<int*>(task_data->outputs[1]);
    auto* output_rows = reinterpret_cast<int*>(task_data->outputs[2]);

    std::vector<double> actualValues(output_values, output_values + expectedValues.size());
    std::vector<int> actualCols(output_cols, output_cols + expectedColIndices.size());
    std::vector<int> actualRows(output_rows, output_rows + expectedRowPtr.size());

    EXPECT_EQ(actualValues, expectedValues);
    EXPECT_EQ(actualCols, expectedColIndices);
    EXPECT_EQ(actualRows, expectedRowPtr);
  }
}

TEST(chernova_n_matrix_multiplication_crs_mpi, test_sparse_14x14_parallel) {
  const int root_rank = 0;
  boost::mpi::communicator world_;

  std::vector<double> valuesA, valuesB;
  std::vector<int> colIndicesA, colIndicesB;
  std::vector<int> rowPtrA, rowPtrB;

  if (world_.rank() == root_rank) {
    valuesA = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0};
    colIndicesA = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0};
    rowPtrA = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

    valuesB = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0};
    colIndicesB = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1};
    rowPtrB = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  }

  std::vector<double> expectedValues = {20.0,  60.0,  120.0,  200.0,  300.0,  420.0,  560.0,
                                        720.0, 900.0, 1100.0, 1320.0, 1560.0, 1820.0, 140.0};
  std::vector<int> expectedColIndices = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2};
  std::vector<int> expectedRowPtr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world_.rank() == root_rank) {
    SetupTaskData(valuesA, colIndicesA, rowPtrA, task_data, world_);
    SetupTaskData(valuesB, colIndicesB, rowPtrB, task_data, world_);

    std::vector<double> resultValues(expectedValues.size());
    std::vector<int> resultColIndices(expectedColIndices.size());
    std::vector<int> resultRowPtr(expectedRowPtr.size());

    SetupOutData(resultValues, resultColIndices, resultRowPtr, task_data, world_);
  }

  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI test_task(task_data);

  if (world_.rank() == root_rank) {
    ASSERT_TRUE(test_task.ValidationImpl());
  }
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world_.rank() == root_rank) {
    auto* output_values = reinterpret_cast<double*>(task_data->outputs[0]);
    auto* output_cols = reinterpret_cast<int*>(task_data->outputs[1]);
    auto* output_rows = reinterpret_cast<int*>(task_data->outputs[2]);

    std::vector<double> actualValues(output_values, output_values + expectedValues.size());
    std::vector<int> actualCols(output_cols, output_cols + expectedColIndices.size());
    std::vector<int> actualRows(output_rows, output_rows + expectedRowPtr.size());

    EXPECT_EQ(actualValues, expectedValues);
    EXPECT_EQ(actualCols, expectedColIndices);
    EXPECT_EQ(actualRows, expectedRowPtr);
  }
}

TEST(chernova_n_matrix_multiplication_crs_mpi, random_10x10) {
  const int matrix_size = 10;
  const double density = 0.1;
  const int root_rank = 0;
  boost::mpi::communicator world_;

  std::vector<double> valuesA, valuesB;
  std::vector<int> colIndicesA, colIndicesB;
  std::vector<int> rowPtrA, rowPtrB;

  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS matrixA;
  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS matrixB;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world_.rank() == root_rank) {
    matrixA = generateRandomCRS(matrix_size, density);
    matrixB = generateIdentityCRS(matrix_size);

    SetupTaskData(matrixA.values, matrixA.col_indices, matrixA.row_ptr, task_data, world_);
    SetupTaskData(matrixB.values, matrixB.col_indices, matrixB.row_ptr, task_data, world_);

    std::vector<double> resultValues(matrixA.values.size());
    std::vector<int> resultColIndices(matrixA.col_indices.size());
    std::vector<int> resultRowPtr(matrixA.row_ptr.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultValues.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultColIndices.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultRowPtr.data()));
    task_data->outputs_count.emplace_back(resultValues.size());
  }

  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI test_task(task_data);

  if (world_.rank() == root_rank) {
    ASSERT_TRUE(test_task.ValidationImpl());
  }
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world_.rank() == root_rank) {
    auto* output_values = reinterpret_cast<double*>(task_data->outputs[0]);
    auto* output_cols = reinterpret_cast<int*>(task_data->outputs[1]);
    auto* output_rows = reinterpret_cast<int*>(task_data->outputs[2]);

    std::vector<double> actualValues(output_values, output_values + matrixA.values.size());
    std::vector<int> actualCols(output_cols, output_cols + matrixA.col_indices.size());
    std::vector<int> actualRows(output_rows, output_rows + matrixA.row_ptr.size());

    EXPECT_EQ(actualValues, matrixA.values);
    EXPECT_EQ(actualCols, matrixA.col_indices);
    EXPECT_EQ(actualRows, matrixA.row_ptr);
  }
}

TEST(chernova_n_matrix_multiplication_crs_mpi, random_20x20) {
  const int matrix_size = 20;
  const double density = 0.1;
  const int root_rank = 0;
  boost::mpi::communicator world_;

  std::vector<double> valuesA, valuesB;
  std::vector<int> colIndicesA, colIndicesB;
  std::vector<int> rowPtrA, rowPtrB;

  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS matrixA;
  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS matrixB;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world_.rank() == root_rank) {
    matrixA = generateRandomCRS(matrix_size, density);
    matrixB = generateIdentityCRS(matrix_size);

    SetupTaskData(matrixA.values, matrixA.col_indices, matrixA.row_ptr, task_data, world_);
    SetupTaskData(matrixB.values, matrixB.col_indices, matrixB.row_ptr, task_data, world_);

    std::vector<double> resultValues(matrixA.values.size());
    std::vector<int> resultColIndices(matrixA.col_indices.size());
    std::vector<int> resultRowPtr(matrixA.row_ptr.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultValues.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultColIndices.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultRowPtr.data()));
    task_data->outputs_count.emplace_back(resultValues.size());
  }

  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI test_task(task_data);

  if (world_.rank() == root_rank) {
    ASSERT_TRUE(test_task.ValidationImpl());
  }
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world_.rank() == root_rank) {
    auto* output_values = reinterpret_cast<double*>(task_data->outputs[0]);
    auto* output_cols = reinterpret_cast<int*>(task_data->outputs[1]);
    auto* output_rows = reinterpret_cast<int*>(task_data->outputs[2]);

    std::vector<double> actualValues(output_values, output_values + matrixA.values.size());
    std::vector<int> actualCols(output_cols, output_cols + matrixA.col_indices.size());
    std::vector<int> actualRows(output_rows, output_rows + matrixA.row_ptr.size());

    EXPECT_EQ(actualValues, matrixA.values);
    EXPECT_EQ(actualCols, matrixA.col_indices);
    EXPECT_EQ(actualRows, matrixA.row_ptr);
  }
}


TEST(chernova_n_matrix_multiplication_crs_mpi, random_1000x1000) {
  const int matrix_size = 1000;
  const double density = 0.1;
  const int root_rank = 0;
  boost::mpi::communicator world_;

  std::vector<double> valuesA, valuesB;
  std::vector<int> colIndicesA, colIndicesB;
  std::vector<int> rowPtrA, rowPtrB;

  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS matrixA;
  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS matrixB;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world_.rank() == root_rank) {
    matrixA = generateRandomCRS(matrix_size, density);
    matrixB = generateIdentityCRS(matrix_size);

    SetupTaskData(matrixA.values, matrixA.col_indices, matrixA.row_ptr, task_data, world_);
    SetupTaskData(matrixB.values, matrixB.col_indices, matrixB.row_ptr, task_data, world_);

    std::vector<double> resultValues(matrixA.values.size());
    std::vector<int> resultColIndices(matrixA.col_indices.size());
    std::vector<int> resultRowPtr(matrixA.row_ptr.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultValues.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultColIndices.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultRowPtr.data()));
    task_data->outputs_count.emplace_back(resultValues.size());
  }

  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI test_task(task_data);

  if (world_.rank() == root_rank) {
    ASSERT_TRUE(test_task.ValidationImpl());
  }
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world_.rank() == root_rank) {
    auto* output_values = reinterpret_cast<double*>(task_data->outputs[0]);
    auto* output_cols = reinterpret_cast<int*>(task_data->outputs[1]);
    auto* output_rows = reinterpret_cast<int*>(task_data->outputs[2]);

    std::vector<double> actualValues(output_values, output_values + matrixA.values.size());
    std::vector<int> actualCols(output_cols, output_cols + matrixA.col_indices.size());
    std::vector<int> actualRows(output_rows, output_rows + matrixA.row_ptr.size());

    EXPECT_EQ(actualValues, matrixA.values);
    EXPECT_EQ(actualCols, matrixA.col_indices);
    EXPECT_EQ(actualRows, matrixA.row_ptr);
  }
}
