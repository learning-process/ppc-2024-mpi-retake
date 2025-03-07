#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/Konstantinov_I_Gauss_Jordan_method/include/ops_seq.hpp"


TEST(Konstantinov_i_gauss_jordan_seq, identity_matrix_3x3) {

  std::vector<double> input_matrix = {1, 0, 0, 5, 0, 1, 0, -3, 0, 0, 1, 8};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.ValidationImpl());
  task_seq.PreProcessingImpl();
  ASSERT_TRUE(task_seq.RunImpl());
  task_seq.PostProcessingImpl();

  std::vector<double> expected_result = {1, 0, 0, 5, 0, 1, 0, -3, 0, 0, 1, 8};
  ASSERT_EQ(output_result, expected_result);
}

TEST(Konstantinov_i_gauss_jordan_seq, zero_pivot_swap) {

  std::vector<double> input_matrix = {0, 1, 2, 9, 1, 0, 3, 8, 4, 5, 6, 30};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.ValidationImpl());
  task_seq.PreProcessingImpl();
  ASSERT_TRUE(task_seq.RunImpl());
  task_seq.PostProcessingImpl();

  std::vector<double> expected_result = {1, 0, 0, -0.8125, 0, 1, 0, 3.125, 0, 0, 1, 2.9375};
  ASSERT_EQ(output_result, expected_result);
}

TEST(Konstantinov_i_gauss_jordan_seq, negative_and_fractional) {

  std::vector<double> input_matrix = {2, -1, 3, 7, -1, 2, -2, -1, 3, -2, 4, 10};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.ValidationImpl());
  task_seq.PreProcessingImpl();
  ASSERT_TRUE(task_seq.RunImpl());
  task_seq.PostProcessingImpl();

  std::vector<double> expected_result = {1, 0, 0, 5, 0, 1, 0, 1.5, 0, 0, 1, -0.5};
  for (size_t i = 0; i < output_result.size(); ++i) {
    ASSERT_TRUE(std::fabs(output_result[i] - expected_result[i]) < 1e-6);
  }
}

TEST(Konstantinov_i_gauss_jordan_seq, two_by_two_simple) {

  std::vector<double> input_matrix = {3, 2, 5, 1, -1, 0};
  int n = 2;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.ValidationImpl());
  task_seq.PreProcessingImpl();
  ASSERT_TRUE(task_seq.RunImpl());
  task_seq.PostProcessingImpl();

  std::vector<double> expected_result = {1, 0, 1, 0, 1, 1};
  for (size_t i = 0; i < output_result.size(); ++i) {
    ASSERT_TRUE(std::fabs(output_result[i] - expected_result[i]) < 1e-6);
  }
}

TEST(Konstantinov_i_gauss_jordan_seq, singular_matrix) {

  std::vector<double> input_matrix = {1, 2, 3, 6, 2, 4, 6, 12, 3, 6, 9, 18};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.ValidationImpl());
  task_seq.PreProcessingImpl();
  EXPECT_FALSE(task_seq.RunImpl());
  task_seq.PostProcessingImpl();
}

TEST(Konstantinov_i_gauss_jordan_seq, four_by_four_matrix) {

  std::vector<double> input_matrix = {2, 1, -1, 2, 3, 1, 0, 2, -1, 4, 3, 2, -1, 1, 5, 1, -1, 1, 0, 2};
  int n = 4;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  task_data_seq->inputs_count.emplace_back(input_matrix.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.ValidationImpl());
  task_seq.PreProcessingImpl();
  ASSERT_TRUE(task_seq.RunImpl());
  task_seq.PostProcessingImpl();

  std::vector<double> expected_result = {1, 0, 0, 0, 13.0 / 9, 0, 1, 0, 0, 8.0 / 9,
                                         0, 0, 1, 0, 13.0 / 9, 0, 0, 0, 1, 1.0 / 3};
  for (size_t i = 0; i < output_result.size(); ++i) {
    ASSERT_TRUE(std::fabs(output_result[i] - expected_result[i]) < 1e-6);
  }
}

TEST(Konstantinov_i_gauss_jordan_seq, three_simple_matrix) {
  std::vector<double> input_matrix = {1, 2, 1, 10, 4, 8, 3, 20, 2, 5, 9, 30};
  int n = 3;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  task_data_seq->inputs_count.emplace_back(input_matrix.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq task_sequential(task_data_seq);
  ASSERT_TRUE(task_sequential.ValidationImpl());
  task_sequential.PreProcessingImpl();
  task_sequential.RunImpl();
  task_sequential.PostProcessingImpl();

  std::vector<double> expected_result = {1, 0, 0, 250, 0, 1, 0, -130, 0, 0, 1, 20};
  ASSERT_EQ(output_result, expected_result);
}

TEST(Konstantinov_i_gauss_jordan_seq, five_simple_matrix_at_1_iter) {
  std::vector<double> input_matrix = {0,  2,  3,  4, 5,  6,  0,  8,  9,  10, 11, 12, 0,  14, 15,
                                      16, 17, 18, 0, 20, 21, 22, 23, 24, 0,  26, 27, 28, 29, 30};
  int n = 5;
  std::vector<double> output_result(n * (n + 1));

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(input_matrix.data())));
  task_data_seq->inputs_count.emplace_back(input_matrix.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq task_sequential(task_data_seq);
  ASSERT_TRUE(task_sequential.ValidationImpl());
  task_sequential.PreProcessingImpl();
  EXPECT_FALSE(task_sequential.RunImpl());
  task_sequential.PostProcessingImpl();
}