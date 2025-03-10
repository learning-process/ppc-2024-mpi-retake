#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/vasenkov_a_gauss_jordan/include/ops_seq.hpp"

TEST(vasenkov_a_gauss_jordan_seq, three_simple_matrix) {
  std::vector<double> input_matrix = {1, 0, 0, 5, 0, 1, 0, -3, 0, 0, 1, 2};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::vector<double> expected_result = {1, 0, 0, 5, 0, 1, 0, -3, 0, 0, 1, 2};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(input_matrix.data())));
  task_data_seq->inputs_count.emplace_back(input_matrix.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  vasenkov_a_gauss_jordan_seq::GaussJordanMethodSequential task_sequential(task_data_seq);
  ASSERT_TRUE(task_sequential.ValidationImpl());
  task_sequential.PreProcessingImpl();
  task_sequential.RunImpl();
  task_sequential.PostProcessingImpl();

  ASSERT_EQ(output_result, expected_result);
}

TEST(vasenkov_a_gauss_jordan_seq, zero_column) {
  std::vector<double> input_matrix = {0, 2, 3, 4, 0, 5, 6, 7, 0, 8, 9, 10};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(input_matrix.data())));
  task_data_seq->inputs_count.emplace_back(input_matrix.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  vasenkov_a_gauss_jordan_seq::GaussJordanMethodSequential task_sequential(task_data_seq);
  ASSERT_TRUE(task_sequential.ValidationImpl());
  task_sequential.PreProcessingImpl();
  EXPECT_FALSE(task_sequential.RunImpl());
  task_sequential.PostProcessingImpl();
}
