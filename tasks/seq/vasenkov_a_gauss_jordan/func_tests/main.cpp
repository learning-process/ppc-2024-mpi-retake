#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/vasenkov_a_gauss_jordan/include/ops_seq.hpp"

TEST(vasenkov_a_gauss_jordan_seq, three_simple_matrix) {
  std::vector<double> input_matrix = {1, 0, 0, 5, 0, 1, 0, -3, 0, 0, 1, 2};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::vector<double> expected_result = {1, 0, 0, 5, 0, 1, 0, -3, 0, 0, 1, 2};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(input_matrix.data())));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasenkov_a_gauss_jordan_seq::GaussJordanMethodSequential taskSequential(taskDataSeq);
  ASSERT_TRUE(taskSequential.ValidationImpl());
  taskSequential.PreProcessingImpl();
  taskSequential.RunImpl();
  taskSequential.PostProcessingImpl();

  ASSERT_EQ(output_result, expected_result);
}

TEST(vasenkov_a_gauss_jordan_seq, zero_column) {
  std::vector<double> input_matrix = {0, 2, 3, 4, 0, 5, 6, 7, 0, 8, 9, 10};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(input_matrix.data())));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  vasenkov_a_gauss_jordan_seq::GaussJordanMethodSequential taskSequential(taskDataSeq);
  ASSERT_TRUE(taskSequential.ValidationImpl());
  taskSequential.PreProcessingImpl();
  EXPECT_FALSE(taskSequential.RunImpl());
  taskSequential.PostProcessingImpl();
}
