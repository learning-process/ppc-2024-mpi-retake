#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/sedova_o_mult_matrices_ccs/include/ops_seq.hpp"

TEST(sedova_o_mult_matrices_ccs, Test_1) {
  std::vector<std::vector<double>> matrix_A = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<std::vector<double>> matrix_B = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> A;
  std::vector<int> row_ind_A;
  std::vector<int> col_ind_A;
  sedova_o_test_task_seq::ConvertToCCS(matrix_A, A, row_ind_A, col_ind_A);
  std::vector<double> B;
  std::vector<int> row_ind_B;
  std::vector<int> col_ind_B;
  sedova_o_test_task_seq::ConvertToCCS(matrix_B, B, row_ind_B, col_ind_B);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> out(matrix_A.size(), std::vector<double>(matrix_B[0].size(), 0));
  sedova_o_test_task_seq::FillData(task_data, matrix_A.size(), matrix_A[0].size(), matrix_B.size(), matrix_B[0].size(),
                                         A, row_ind_A, col_ind_A, B, row_ind_B, col_ind_B, out);
  sedova_o_test_task_seq::TestTaskSequential testTaskSequential(task_data);
  ASSERT_TRUE(testTaskSequential.ValidationImpl());
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  std::vector<std::vector<double>> ans(matrix_A.size(), std::vector<double>(matrix_B[0].size(), 0));
  for (size_t i = 0; i < out.size(); ++i) {
    auto *ptr = reinterpret_cast<double *>(taskData->outputs[i]);
    ans[i] = std::vector(ptr, ptr + matrix_B.size());
  }
  std::vector<std::vector<double>> check_result = {{2, 0, 6}, {0, 14, 0}, {4, 0, 12}};
  ASSERT_EQ(check_result, ans);
}