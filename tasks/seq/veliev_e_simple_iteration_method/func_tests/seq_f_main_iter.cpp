// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/veliev_e_simple_iteration_method/include/seq_header_iter.hpp"

TEST(veliev_e_simple_iteration_method_seq, veliev_slae_2x2) {
  const int input_size = 2;

  std::vector<double> matrix = {4, 1, 1, 3};

  std::vector<double> g = {9, 5};

  std::vector<double> x(input_size, 0.0);

  std::vector<double> expected_solution = {2, 1};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.push_back(input_size);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(g.data()));
  task_data_seq->inputs_count.push_back(input_size);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(input_size);

  veliev_e_simple_iteration_method_seq::VelievSlaeIterSeq test1(task_data_seq);

  ASSERT_TRUE(test1.ValidationImpl());
  test1.PreProcessingImpl();
  test1.RunImpl();
  test1.PostProcessingImpl();

  for (int i = 0; i < input_size; ++i) {
    EXPECT_NEAR(x[i], expected_solution[i], 1e-6);
  }
}

TEST(veliev_e_simple_iteration_method_seq, veliev_slae_non_dominant_matrix) {
  const int input_size = 3;

  std::vector<double> matrix = {3.0, -1.0, 2.0, -1.0, 5.0, -2.0, 2.0, -2.0, 4.0, 0.5, 1.5, -1.0};
  std::vector<double> b = {10.0, 12.0, 9.0};
  std::vector<double> x(input_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.push_back(4);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.push_back(input_size);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(input_size);

  veliev_e_simple_iteration_method_seq::VelievSlaeIterSeq test1(task_data_seq);

  ASSERT_FALSE(test1.ValidationImpl());
}

TEST(veliev_e_simple_iteration_method_seq, VelievSlaeIterSeq_IterationConvergence_3x3) {
  const int input_size = 3;

  std::vector<double> matrix = {10, -1, 2, -1, 11, -1, 2, -1, 10};

  std::vector<double> g = {6, 25, -11};

  std::vector<double> x(input_size, 0.0);

  std::vector<double> expected_solution = {217.0 / 208.0, 59.0 / 26.0, -225.0 / 208.0};
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.push_back(input_size);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(g.data()));
  task_data_seq->inputs_count.push_back(input_size);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(input_size);

  veliev_e_simple_iteration_method_seq::VelievSlaeIterSeq test1(task_data_seq);

  ASSERT_TRUE(test1.ValidationImpl());
  test1.PreProcessingImpl();
  test1.RunImpl();
  test1.PostProcessingImpl();

  for (int i = 0; i < input_size; ++i) {
    EXPECT_NEAR(x[i], expected_solution[i], 1e-6);
  }
}

TEST(veliev_e_simple_iteration_method_seq, veliev_slae_4x4) {
  const int input_size = 4;

  std::vector<double> matrix = {10, 1, 2, 4, 1, 9, 5, 2, 4, 1, 10, -2, 5, 1, 1, 23};

  std::vector<double> g = {7, 100, 9, 7};

  std::vector<double> x(input_size, 0.0);

  std::vector<double> expected_solution = {-2982.0 / 8215, 92239.0 / 8215, -803.0 / 8215, -827.0 / 8215};
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.push_back(input_size);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(g.data()));
  task_data_seq->inputs_count.push_back(input_size);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(input_size);

  veliev_e_simple_iteration_method_seq::VelievSlaeIterSeq test1(task_data_seq);

  ASSERT_TRUE(test1.ValidationImpl());
  test1.PreProcessingImpl();
  test1.RunImpl();
  test1.PostProcessingImpl();

  for (int i = 0; i < input_size; ++i) {
    EXPECT_NEAR(x[i], expected_solution[i], 1e-6);
  }
}