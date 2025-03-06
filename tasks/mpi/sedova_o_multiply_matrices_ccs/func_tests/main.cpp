#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/task/include/task.hpp"
#include "mpi/sedova_o_multiply_matrices_ccs/include/ops_mpi.hpp"

namespace sedova_o_multiply_matrices_ccs_mpi {
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
  int rows_a = static_cast<int>(a.size());
  int cols_a = static_cast<int>(a[0].size());
  int cols_b = static_cast<int>(b[0].size());
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

void FuncTestTemplate(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b) {  // NOLINT
  boost::mpi::communicator world;
  std::vector<double> a_val;
  std::vector<int> a_row_ind;
  std::vector<int> a_col_ptr;
  int rows_a = a.size();     //  NOLINT
  int cols_a = a[0].size();  //  NOLINT

  std::vector<double> b_val;
  std::vector<int> b_row_ind;
  std::vector<int> b_col_ptr;
  int rows_b = b.size());     //  NOLINT
  int cols_b = b[0].size();  //  NOLINT

  std::vector<double> exp_c_val;
  std::vector<int> exp_c_row_ind;
  std::vector<int> exp_c_col_ptr;

  if (world.rank() == 0) {                                                                                //  NOLINT
    auto exp_c = sedova_o_multiply_matrices_ccs_mpi::MultiplyMatrices(a, b);                              //  NOLINT
    sedova_o_multiply_matrices_ccs_mpi::Convertirovanie(exp_c, exp_c.size(), exp_c[0].size(), exp_c_val,  //  NOLINT
                                                        exp_c_row_ind, exp_c_col_ptr);
  }
  std::vector<double> c_val;
  std::vector<int> c_row_ind;
  std::vector<int> c_col_ptr;

  sedova_o_multiply_matrices_ccs_mpi::Convertirovanie(a, rows_a, cols_a, a_val, a_row_ind, a_col_ptr);  //  NOLINT
  sedova_o_multiply_matrices_ccs_mpi::Convertirovanie(b, rows_b, cols_b, b_val, b_row_ind, b_col_ptr);  //  NOLINT

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();  //  NOLINT
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_a));                      //  NOLINT
  task_data->inputs_count.emplace_back(1);                                                   //  NOLINT
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_a));                      //  NOLINT
  task_data->inputs_count.emplace_back(1);                                                   //  NOLINT
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_b));                      //  NOLINT
  task_data->inputs_count.emplace_back(1);                                                   //  NOLINT
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_b));                      //  NOLINT
  task_data->inputs_count.emplace_back(1);                                                   //  NOLINT

  if (world.rank() == 0) {                                                          //  NOLINT
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_val.data()));      //  NOLINT
    task_data->inputs_count.emplace_back(a_val.size());                             //  NOLINT
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_row_ind.data()));  //  NOLINT
    task_data->inputs_count.emplace_back(a_row_ind.size());                         //  NOLINT
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col_ptr.data()));  //  NOLINT
    task_data->inputs_count.emplace_back(a_col_ptr.size());                         //  NOLINT

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_val.data()));      //  NOLINT
    task_data->inputs_count.emplace_back(b_val.size());                             //  NOLINT
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_row_ind.data()));  //  NOLINT
    task_data->inputs_count.emplace_back(b_row_ind.size());                         //  NOLINT
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col_ptr.data()));  //  NOLINT
    task_data->inputs_count.emplace_back(b_col_ptr.size());                         //  NOLINT

    c_val.resize(exp_c_val.size());          //  NOLINT
    c_row_ind.resize(exp_c_row_ind.size());  //  NOLINT
    c_col_ptr.resize(exp_c_col_ptr.size());  //  NOLINT

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_val.data()));      //  NOLINT
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_row_ind.data()));  //  NOLINT
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c_col_ptr.data()));  //  NOLINT
  }

  sedova_o_multiply_matrices_ccs_mpi::TestTaskMPI task(task_data);  //  NOLINT
  bool validation_impl = task.ValidationImpl();                     //  NOLINT
  boost::mpi::broadcast(world, validation_impl, 0);                 //  NOLINT
  if (validation_impl) {                                            //  NOLINT
    task.PreProcessingImpl();                                       //  NOLINT
    task.RunImpl();                                                 //  NOLINT
    task.PostProcessingImpl();                                      //  NOLINT
    if (world.rank() == 0) {                                        //  NOLINT
      ASSERT_EQ(exp_c_val, c_val);                                  //  NOLINT
      ASSERT_EQ(exp_c_row_ind, c_row_ind);                          //  NOLINT
      ASSERT_EQ(exp_c_col_ptr, c_col_ptr);                          //  NOLINT
    }
  }
}
}  // namespace
}  // namespace sedova_o_multiply_matrices_ccs_mpi

TEST(sedova_o_multiply_matrices_ccs_mpi, SmallMatrices) {
  std::vector<std::vector<double>> a = {{1, 0, 2}, {0, 3, 0}};
  std::vector<std::vector<double>> b = {{0, 4, 0, 0, 1}, {5, 0, 0, 2, 0}, {0, 0, 3, 0, 6}};
  sedova_o_multiply_matrices_ccs_mpi::FuncTestTemplate(a, b);  //  NOLINT
}

TEST(sedova_o_multiply_matrices_ccs_mpi, Random3x5And5x4) {
  auto a = sedova_o_multiply_matrices_ccs_mpi::GenerateMatrix(3, 5, 5);
  auto b = sedova_o_multiply_matrices_ccs_mpi::GenerateMatrix(5, 4, 10);
  sedova_o_multiply_matrices_ccs_mpi::FuncTestTemplate(a, b);  //  NOLINT
}
