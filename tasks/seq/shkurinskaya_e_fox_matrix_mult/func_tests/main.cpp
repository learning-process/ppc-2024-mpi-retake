#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shkurinskaya_e_fox_matrix_mult/include/ops_sec.hpp"

TEST(shkurinskaya_e_fox_mat_mul_seq, small_matrix) {
  int matrix_size = 2;
  std::vector<double> in1 = shkurinskaya_e_fox_mat_mul_seq::GetRandomMatrix(matrix_size, matrix_size);
  std::vector<double> in2 = shkurinskaya_e_fox_mat_mul_seq::GetRandomMatrix(matrix_size, matrix_size);
  std::vector<double> out(matrix_size * matrix_size), ans(matrix_size * matrix_size, 0.0);

  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      for (int k = 0; k < matrix_size; ++k) {
        ans[(i * matrix_size) + j] += in1[(i * matrix_size) + k] * in2[(k * matrix_size) + j];
      }
    }
  }

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(matrix_size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(matrix_size);

  // Create Task
  shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (int it = 0; it < (int)ans.size(); ++it) {
    ASSERT_NEAR(ans[it], out[it], 1);
  }
}

TEST(shkurinskaya_e_fox_mat_mul_seq, not_small_matrix) {
  int matrix_size = 20;
  std::vector<double> in1 = shkurinskaya_e_fox_mat_mul_seq::GetRandomMatrix(matrix_size, matrix_size);
  std::vector<double> in2 = shkurinskaya_e_fox_mat_mul_seq::GetRandomMatrix(matrix_size, matrix_size);
  std::vector<double> out(matrix_size * matrix_size), ans(matrix_size * matrix_size, 0.0);

  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      for (int k = 0; k < matrix_size; ++k) {
        ans[(i * matrix_size) + j] += in1[(i * matrix_size) + k] * in2[(k * matrix_size) + j];
      }
    }
  }

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(matrix_size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(matrix_size);

  // Create Task
  shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (int it = 0; it < (int)ans.size(); ++it) {
    ASSERT_NEAR(ans[it], out[it], 1);
  }
}

TEST(shkurinskaya_e_fox_mat_mul_seq, big_matrix) {
  int matrix_size = 100;
  std::vector<double> in1 = shkurinskaya_e_fox_mat_mul_seq::GetRandomMatrix(matrix_size, matrix_size);
  std::vector<double> in2 = shkurinskaya_e_fox_mat_mul_seq::GetRandomMatrix(matrix_size, matrix_size);
  std::vector<double> out(matrix_size * matrix_size), ans(matrix_size * matrix_size, 0.0);

  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      for (int k = 0; k < matrix_size; ++k) {
        ans[(i * matrix_size) + j] += in1[(i * matrix_size) + k] * in2[(k * matrix_size) + j];
      }
    }
  }

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(matrix_size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(matrix_size);

  // Create Task
  shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (int it = 0; it < (int)ans.size(); ++it) {
    ASSERT_NEAR(ans[it], out[it], 1);
  }
}

TEST(shkurinskaya_e_fox_mat_mul_seq, validation_false_1) {
  int matrix_size = 0;
  std::vector<double> in1;
  std::vector<double> in2;
  std::vector<double> out;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(matrix_size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(matrix_size);

  // Create Task
  shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(shkurinskaya_e_fox_mat_mul_seq, validation_false_2) {
  int matrix_size = 1;
  std::vector<double> in1;
  std::vector<double> in2;
  std::vector<double> out;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(matrix_size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  shkurinskaya_e_fox_mat_mul_seq::FoxMatMulSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
