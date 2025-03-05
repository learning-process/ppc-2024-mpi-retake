#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/sedova_o_multiply_matrices_ccs/include/ops_mpi.hpp"

namespace sedova_o_multiply_matrices_ccs_mpi {

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
  int rows_A = a.size();
  int cols_A = a[0].size();
  int cols_B = b[0].size();

  std::vector<std::vector<double>> result(rows_A, std::vector<double>(cols_B, 0.0));

  for (int i = 0; i < rows_A; ++i) {
    for (int j = 0; j < cols_B; ++j) {
      for (int k = 0; k < cols_A; ++k) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}

}  // namespace sedova_o_multiply_matrices_ccs_mpi

TEST(sedova_o_multiply_matrices_ccs_mpi, SmallMatrices) {
  std::vector<std::vector<double>> a_ = {{1, 0, 2}, {0, 3, 0}};
  std::vector<std::vector<double>> b_ = {{0, 4, 0, 0, 1}, {5, 0, 0, 2, 0}, {0, 0, 3, 0, 6}};
  boost::mpi::communicator world;
  std::vector<double> A_val_;
  std::vector<int> A_row_ind_;
  std::vector<int> A_col_ptr_;
  int rows_A_ = a_.size();
  int cols_A_ = a_[0].size();

  std::vector<double> B_val_;
  std::vector<int> B_row_ind_;
  std::vector<int> B_col_ptr_;
  int rows_B_ = b_.size();
  int cols_B_ = b_[0].size();

  std::vector<double> exp_C_val;
  std::vector<int> exp_C_row_ind;
  std::vector<int> exp_C_col_ptr;

  if (world.rank() == 0) {
    auto exp_C = MultiplyMatrices(a_, b_);
    Convertirovanie(exp_C, exp_C.size(), exp_C[0].size(), exp_C_val, exp_C_row_ind, exp_C_col_ptr);
  }
  std::vector<double> C_val;
  std::vector<int> C_row_ind;
  std::vector<int> C_col_ptr;

  Convertirovanie(a_, rows_A_, cols_A_, A_val_, A_row_ind_, A_col_ptr_);
  Convertirovanie(b_, rows_B_, cols_B_, B_val_, B_row_ind_, B_col_ptr_);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_A_));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_A_));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_B_));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_B_));
  task_data->inputs_count.emplace_back(1);
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_val_.data()));
    task_data->inputs_count.emplace_back(A_val_.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_row_ind_.data()));
    task_data->inputs_count.emplace_back(A_row_ind_.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col_ptr_.data()));
    task_data->inputs_count.emplace_back(A_col_ptr_.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_val_.data()));
    task_data->inputs_count.emplace_back(B_val_.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_row_ind_.data()));
    task_data->inputs_count.emplace_back(B_row_ind_.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col_ptr_.data()));
    task_data->inputs_count.emplace_back(B_col_ptr_.size());

    C_val.resize(exp_C_val.size());
    C_row_ind.resize(exp_C_row_ind.size());
    C_col_ptr.resize(exp_C_col_ptr.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_val.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_ind.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));
  }

  TestTaskMPI task(task_data);
  bool ValidationImpl = task.ValidationImpl();
  boost::mpi::broadcast(world, ValidationImpl, 0);
  if (ValidationImpl) {
    task.PreProcessingImpl();
    task.RunImpl();
    task.PostProcessingImpl();
    if (world.rank() == 0) {
      ASSERT_EQ(exp_C_val, C_val);
      ASSERT_EQ(exp_C_row_ind, C_row_ind);
      ASSERT_EQ(exp_C_col_ptr, C_col_ptr);
    }
  }
}

TEST(sedova_o_multiply_matrices_ccs_mpi, Random3x5And5x4) {
  auto a_ = sedova_o_multiply_matrices_ccs_mpi::GenerateMatrix(3, 5, 5);
  auto b_ = sedova_o_multiply_matrices_ccs_mpi::GenerateMatrix(5, 4, 10);
  boost::mpi::communicator world;
  std::vector<double> A_val;
  std::vector<int> A_row_ind;
  std::vector<int> A_col_ptr;
  int rows_A = a_.size();
  int cols_A = a_[0].size();

  std::vector<double> B_val;
  std::vector<int> B_row_ind;
  std::vector<int> B_col_ptr;
  int rows_B = b_.size();
  int cols_B = b_[0].size();

  std::vector<double> exp_C_val;
  std::vector<int> exp_C_row_ind;
  std::vector<int> exp_C_col_ptr;

  if (world.rank() == 0) {
    auto exp_C = MultiplyMatrices(a_, b_);
    Convertirovanie(exp_C, exp_C.size(), exp_C[0].size(), exp_C_val, exp_C_row_ind, exp_C_col_ptr);
  }
  std::vector<double> C_val;
  std::vector<int> C_row_ind;
  std::vector<int> C_col_ptr;

  Convertirovanie(a_, rows_A, cols_A, A_val, A_row_ind, A_col_ptr);
  Convertirovanie(b_, rows_B, cols_B, B_val, B_row_ind, B_col_ptr);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_A));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_A));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_B));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_B));
  task_data->inputs_count.emplace_back(1);
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_val.data()));
    task_data->inputs_count.emplace_back(A_val.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_row_ind.data()));
    task_data->inputs_count.emplace_back(A_row_ind.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col_ptr.data()));
    task_data->inputs_count.emplace_back(A_col_ptr.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_val.data()));
    task_data->inputs_count.emplace_back(B_val.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_row_ind.data()));
    task_data->inputs_count.emplace_back(B_row_ind.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col_ptr.data()));
    task_data->inputs_count.emplace_back(B_col_ptr.size());

    C_val.resize(exp_C_val.size());
    C_row_ind.resize(exp_C_row_ind.size());
    C_col_ptr.resize(exp_C_col_ptr.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_val.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_ind.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));
  }

  TestTaskMPI task(task_data);
  bool ValidationImpl = task.ValidationImpl();
  boost::mpi::broadcast(world, ValidationImpl, 0);
  if (ValidationImpl) {
    task.PreProcessingImpl();
    task.RunImpl();
    task.PostProcessingImpl();
    if (world.rank() == 0) {
      ASSERT_EQ(exp_C_val, C_val);
      ASSERT_EQ(exp_C_row_ind, C_row_ind);
      ASSERT_EQ(exp_C_col_ptr, C_col_ptr);
    }
  }
}