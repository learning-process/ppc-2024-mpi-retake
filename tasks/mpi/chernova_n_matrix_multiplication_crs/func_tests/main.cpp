#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/chernova_n_matrix_multiplication_crs/include/ops_mpi.hpp"

namespace {
chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS GenerateRandomCrs(int size, double density) {
  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS matrix;
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> value_dist(1.0, 10.0);

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
  std::cout << std::endl << " Values" << std::endl;
  for (int i = 0; i < matrix.values.size(); i++) {
    std::cout << matrix.values[i] << " ";
  }
  std::cout << std::endl << " col" << std::endl;
  for (int i = 0; i < matrix.col_indices.size(); i++) {
    std::cout << matrix.col_indices[i] << " ";
  }
  std::cout << std::endl << " row" << std::endl;
  for (int i = 0; i < matrix.row_ptr.size(); i++) {
    std::cout << matrix.row_ptr[i] << " ";
  }
  return matrix;
}
chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS GenerateIdentityCrs(int n) {
  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS matrix;
  if (n <= 0) {
    return matrix;
  }
  matrix.values = std::vector<double>(n, 1.0);
  matrix.col_indices = std::vector<int>(n);
  matrix.row_ptr = std::vector<int>(n + 1);

  for (int i = 0; i < n; ++i) {
    matrix.col_indices[i] = i;
  }

  for (int i = 0; i <= n; ++i) {
    matrix.row_ptr[i] = i;
  }

  return matrix;
}

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
void Execution(auto& test_task, const auto& world) {
  if (world.rank() == 0) {
    ASSERT_TRUE(test_task.ValidationImpl());
  }
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();
}
}  // namespace
TEST(chernova_n_matrix_multiplication_crs_mpi, test_sparse_10x10_parallel) {
  boost::mpi::communicator world;

  std::vector<double> values_a;
  std::vector<int> col_indices_a;
  std::vector<int> row_ptr_a;
  std::vector<double> values_b;
  std::vector<int> col_indices_b;
  std::vector<int> row_ptr_b;
  std::vector<double> result_values;
  std::vector<int> result_col_indices;
  std::vector<int> result_row_ptr;

  if (world.rank() == 0) {
    values_a = {3.0, 7.0, 2.0, 5.0, 6.0, 4.0, 9.0, 1.0, 8.0, 10.0};
    col_indices_a = {2, 5, 9, 1, 4, 7, 0, 3, 6, 8};
    row_ptr_a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    values_b = {2.0, 5.0, 1.0, 3.0, 6.0, 8.0, 4.0, 7.0, 9.0, 2.0};
    col_indices_b = {4, 7, 3, 9, 0, 2, 5, 1, 6, 8};
    row_ptr_b = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  }

  std::vector<double> expected_values = {3.0, 56.0, 4.0, 25.0, 36.0, 28.0, 18.0, 3.0, 32.0, 90.0};
  std::vector<int> expected_col_indices = {3, 2, 8, 7, 0, 1, 4, 9, 5, 6};
  std::vector<int> expected_row_ptr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    SetupTaskData(values_a, col_indices_a, row_ptr_a, task_data, world);
    SetupTaskData(values_b, col_indices_b, row_ptr_b, task_data, world);

    result_values = std::vector<double>(expected_values.size());
    result_col_indices = std::vector<int>(expected_col_indices.size());
    result_row_ptr = std::vector<int>(expected_row_ptr.size());

    SetupOutData(result_values, result_col_indices, result_row_ptr, task_data, world);
  }

  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI test_task(task_data);

  Execution(test_task, world);

  if (world.rank() == 0) {
      EXPECT_EQ(result_values, expected_values);
      EXPECT_EQ(result_col_indices, expected_col_indices);
      EXPECT_EQ(result_row_ptr, expected_row_ptr);
  }
}

TEST(chernova_n_matrix_multiplication_crs_mpi, test_sparse_14x14_parallel) {
  boost::mpi::communicator world;

  std::vector<double> values_a;
  std::vector<int> col_indices_a;
  std::vector<int> row_ptr_a;
  std::vector<double> values_b;
  std::vector<int> col_indices_b;
  std::vector<int> row_ptr_b;
  std::vector<double> result_values;
  std::vector<int> result_col_indices;
  std::vector<int> result_row_ptr;

  if (world.rank() == 0) {
    values_a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0};
    col_indices_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0};
    row_ptr_a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

    values_b = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0};
    col_indices_b = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1};
    row_ptr_b = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  }

  std::vector<double> expected_values = {20.0,  60.0,  120.0,  200.0,  300.0,  420.0,  560.0,
                                         720.0, 900.0, 1100.0, 1320.0, 1560.0, 1820.0, 140.0};
  std::vector<int> expected_col_indices = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2};
  std::vector<int> expected_row_ptr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    SetupTaskData(values_a, col_indices_a, row_ptr_a, task_data, world);
    SetupTaskData(values_b, col_indices_b, row_ptr_b, task_data, world);

    result_values = std::vector<double>(expected_values.size());
    result_col_indices = std::vector<int>(expected_col_indices.size());
    result_row_ptr = std::vector<int>(expected_row_ptr.size());

    SetupOutData(result_values, result_col_indices, result_row_ptr, task_data, world);
  }

  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI test_task(task_data);

  Execution(test_task, world);

  if (world.rank() == 0) {
    EXPECT_EQ(result_values, expected_values);
    EXPECT_EQ(result_col_indices, expected_col_indices);
    EXPECT_EQ(result_row_ptr, expected_row_ptr);
  }
}

TEST(chernova_n_matrix_multiplication_crs_mpi, random_10x10) {
  const int matrix_size = 10;
  const double density = 0.1;
  boost::mpi::communicator world;

  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS matrix_a;
  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI::SparseMatrixCRS matrix_b;

  std::vector<double> result_values;
  std::vector<int> result_col_indices;
  std::vector<int> result_row_ptr;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix_a = GenerateRandomCrs(matrix_size, density);
    matrix_b = GenerateIdentityCrs(matrix_size);

    SetupTaskData(matrix_a.values, matrix_a.col_indices, matrix_a.row_ptr, task_data, world);
    SetupTaskData(matrix_b.values, matrix_b.col_indices, matrix_b.row_ptr, task_data, world);

    result_values = std::vector<double>(matrix_a.values.size());
    result_col_indices = std::vector<int>(matrix_a.col_indices.size());
    result_row_ptr = std::vector<int>(matrix_a.row_ptr.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_values.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_col_indices.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_row_ptr.data()));
    task_data->outputs_count.emplace_back(result_values.size());
  }

  chernova_n_matrix_multiplication_crs_mpi::TestTaskMPI test_task(task_data);
  Execution(test_task, world);
  if (world.rank() == 0) {
    EXPECT_EQ(result_values, matrix_a.values);
    EXPECT_EQ(result_col_indices, matrix_a.col_indices);
    EXPECT_EQ(result_row_ptr, matrix_a.row_ptr);
  }
}
