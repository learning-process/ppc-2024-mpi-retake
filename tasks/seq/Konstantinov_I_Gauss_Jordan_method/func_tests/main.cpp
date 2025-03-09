#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/Konstantinov_I_Gauss_Jordan_method/include/ops_seq.hpp"

TEST(Konstantinov_i_gauss_jordan_method_seq, Test_2x2) {
  int n = 2;
  std::vector<double> in = {2, 3, 5, 4, 1, 6};
  std::vector<double> expected_output = {1.3, 0.8};
  std::vector<double> out(n, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size() / (n + 1));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq gauss_task_sequential(task_data_seq);

  ASSERT_TRUE(gauss_task_sequential.ValidationImpl());
  gauss_task_sequential.PreProcessingImpl();
  gauss_task_sequential.RunImpl();
  gauss_task_sequential.PostProcessingImpl();

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(out[i], expected_output[i], 1e-3);
  }
}

TEST(Konstantinov_i_gauss_jordan_method_seq, Test_Gauss_3x3) {
  int n = 3;

  std::vector<double> in = {1, 1, 1, 6, 0, 2, 5, -4, 2, 5, -1, 27};
  std::vector<double> expected_output = {5.0, 3.0, -2.0};
  std::vector<double> out(n, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size() / (n + 1));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq gauss_task_sequential(task_data_seq);

  ASSERT_EQ(gauss_task_sequential.ValidationImpl(), true);
  gauss_task_sequential.PreProcessingImpl();
  gauss_task_sequential.RunImpl();
  gauss_task_sequential.PostProcessingImpl();

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(out[i], expected_output[i], 1e-3);
  }
}

TEST(Konstantinov_i_gauss_jordan_method_seq, Test_Gauss_5x5) {
  int n = 5;

  std::vector<double> in = {2, 3,  -1, 5, 1, 8, 4, -2, 3, -1, 2, 10, -1, 5, 2,
                            3, -4, -3, 3, 2, 4, 1, -2, 6, 1,  1, 1,  1,  1, 4};
  std::vector<double> expected_output = {2.0, 0.142, 0.285, 0.571, 1.0};
  std::vector<double> out(n, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size() / (n + 1));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq gauss_task_sequential(task_data_seq);

  ASSERT_TRUE(gauss_task_sequential.ValidationImpl());
  gauss_task_sequential.PreProcessingImpl();
  gauss_task_sequential.RunImpl();
  gauss_task_sequential.PostProcessingImpl();

  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(out[i], expected_output[i], 1e-3);
  }
}

TEST(Konstantinov_i_gauss_jordan_method_seq, Test_Gauss_Invalid_Data) {
  int n = 2;

  std::vector<double> in = {1, 2, 5};
  std::vector<double> out(n, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size() / (n + 1));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq gauss_task_sequential(task_data_seq);

  ASSERT_EQ(gauss_task_sequential.ValidationImpl(), false);
}

TEST(Konstantinov_i_gauss_jordan_method_seq, Test_Gauss_Zero_Diag) {
  int n = 3;
  std::vector<double> in = {0, 1, 1, 1, 2, 1, 2, 2, 2, 2, 4, 3};
  std::vector<double> out(n, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size() / (n + 1));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq gauss_task_sequential(task_data_seq);

  ASSERT_FALSE(gauss_task_sequential.ValidationImpl());
}

TEST(Konstantinov_i_gauss_jordan_method_seq, Test_Gauss_Overdetermined_System) {
  int n = 3;

  std::vector<double> in = {1, 1, 1, 6, 0, 2, 5, -4, 2, 5, -1, 27, 1, 1, 1, 6};
  std::vector<double> out(n, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size() / (n + 1));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq gauss_task_sequential(task_data_seq);

  ASSERT_FALSE(gauss_task_sequential.ValidationImpl());
}