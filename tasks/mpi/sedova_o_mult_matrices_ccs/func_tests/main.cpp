#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdlib>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/sedova_o_mult_matrices_ccs/include/ops_mpi.hpp"

TEST(sedova_o_test_task_mpi, Test_1) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrix_A = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  std::vector<std::vector<double>> matrix_B = {{2, 0}, {0, 3}, {10, 4}};
  std::vector<double> A;
  std::vector<int> row_ind_A;
  std::vector<int> col_ind_A;
  sedova_o_test_task_mpi::ConvertToCCS(matrix_A, A, row_ind_A, col_ind_A);
  std::vector<double> B;
  std::vector<int> row_ind_B;
  std::vector<int> col_ind_B;
  sedova_o_test_task_mpi::ConvertToCCS(matrix_B, B, row_ind_B, col_ind_B);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> out_par(matrix_A.size(), std::vector<double>(matrix_B[0].size(), 0));
  std::vector<std::vector<double>> out_seq(matrix_A.size(), std::vector<double>(matrix_B[0].size(), 0));
  if (world.rank() == 0) {
    sedova_o_test_task_mpi::FillData(task_data_par, matrix_A.size(), matrix_A[0].size(), matrix_B.size(),
                                     matrix_B[0].size(), A, row_ind_A, col_ind_A, B, row_ind_B, col_ind_B, out_par);
    sedova_o_test_task_mpi::FillData(task_data_seq, matrix_A.size(), matrix_A[0].size(), matrix_B.size(),
                                     matrix_B[0].size(), A, row_ind_A, col_ind_A, B, row_ind_B, col_ind_B, out_seq);
  }
  sedova_o_test_task_mpi::TestTaskMPI TestMpiTaskParallel(task_data_par);
  ASSERT_EQ(TestMpiTaskParallel.ValidationImpl(), true);
  TestMpiTaskParallel.PreProcessingImpl();
  TestMpiTaskParallel.RunImpl();
  TestMpiTaskParallel.PostProcessingImpl();
  if (world.rank() == 0) {
    std::vector<std::vector<double>> ans_par(matrix_A.size(), std::vector<double>(matrix_B[0].size(), 0));
    std::vector<std::vector<double>> ans_seq(matrix_A.size(), std::vector<double>(matrix_B[0].size(), 0));
    for (size_t i = 0; i < out_par.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(task_data_par->outputs[i]);
      ans_par[i] = std::vector(ptr, ptr + matrix_B[0].size());
    }
    sedova_o_test_task_mpi::TestTaskSequential TestMpiTaskSequential(task_data_seq);
    ASSERT_EQ(TestMpiTaskSequential.ValidationImpl(), true);
    TestMpiTaskSequential.PreProcessingImpl();
    TestMpiTaskSequential.RunImpl();
    TestMpiTaskSequential.PostProcessingImpl();
    for (size_t i = 0; i < out_seq.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(task_data_seq->outputs[i]);
      ans_seq[i] = std::vector(ptr, ptr + matrix_B[0].size());
    }
    ASSERT_EQ(ans_seq, ans_par);
  }
}