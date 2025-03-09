#include <gtest/gtest.h>
#include <iostream>
#include "seq/agafeev_s_strassen_alg/include/strassen_seq.hpp"

static std::vector<double> matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, int rowColSize) {
  std::vector<double> C(rowColSize * rowColSize, 0);

  for (int i = 0; i < rowColSize; ++i) {
    for (int j = 0; j < rowColSize; ++j) {
      for (int k = 0; k < rowColSize; ++k) {
        C[i * rowColSize + j] += A[i * rowColSize + k] * B[k * rowColSize + j];
      }
    }
  }

  return C;
}

static std::vector<double> create_RandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(time(nullptr));
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  std::vector<double> matrix(row_size * column_size);
  for (unsigned int i = 0; i < matrix.size(); i++) matrix[i] = dist(rand_gen);

  return matrix;
}

TEST(agafeev_s_strassen_alg_seq, matmul_1x1) {
  const int rows = 1;
  const int columns = 1;

  // Create data
  std::vector<double> in_matrix1 = create_RandomMatrix(rows, columns);
  std::vector<double> in_matrix2 = create_RandomMatrix(rows, columns);
  std::vector<double> out(rows*columns, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(columns);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(columns);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_strassen_alg_seq::MultiplMatrixSequental testTask(task_data);
  bool isValid = testTask.ValidationImpl();
  ASSERT_EQ(isValid, true);
  testTask.PreProcessingImpl();
  testTask.RunImpl();
  testTask.PostProcessingImpl();
  auto right_answer = matrixMultiply(in_matrix1, in_matrix2, rows);

  for (size_t i = 0; i < out.size(); i++) { 
    ASSERT_FLOAT_EQ(right_answer[i], out[i]);
  }
}

TEST(agafeev_s_strassen_alg_seq, matmul_2x2) {
  const int rows = 2;
  const int columns = 2;

  // Create data
  std::vector<double> in_matrix1 = create_RandomMatrix(rows, columns);
  std::vector<double> in_matrix2 = create_RandomMatrix(rows, columns);
  std::vector<double> out(rows*columns, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(columns);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(columns);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_strassen_alg_seq::MultiplMatrixSequental testTask(task_data);
  bool isValid = testTask.ValidationImpl();
  ASSERT_EQ(isValid, true);
  testTask.PreProcessingImpl();
  testTask.RunImpl();
  testTask.PostProcessingImpl();
  auto right_answer = matrixMultiply(in_matrix1, in_matrix2, rows);

  for (size_t i = 0; i < out.size(); i++) { 
    ASSERT_FLOAT_EQ(right_answer[i], out[i]);
  }
}

TEST(agafeev_s_strassen_alg_seq, matmul_64x64) {
  const int rows = 2;
  const int columns = 2;

  // Create data
  std::vector<double> in_matrix1 = create_RandomMatrix(rows, columns);
  std::vector<double> in_matrix2 = create_RandomMatrix(rows, columns);
  std::vector<double> out(rows*columns, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(columns);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(columns);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_strassen_alg_seq::MultiplMatrixSequental testTask(task_data);
  bool isValid = testTask.ValidationImpl();
  ASSERT_EQ(isValid, true);
  testTask.PreProcessingImpl();
  testTask.RunImpl();
  testTask.PostProcessingImpl();
  auto right_answer = matrixMultiply(in_matrix1, in_matrix2, rows);

  for (size_t i = 0; i < out.size(); i++) { 
    ASSERT_FLOAT_EQ(right_answer[i], out[i]);
  }
}

TEST(agafeev_s_strassen_alg_seq, wrong_rowcolumn_valid) {
  // Create data
  std::vector<double> in_matrix1 = create_RandomMatrix(3, 5);
  std::vector<double> in_matrix2 = create_RandomMatrix(5, 4);
  std::vector<double> out(3*4, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix1.data()));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs_count.emplace_back(5);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_matrix2.data()));
  task_data->inputs_count.emplace_back(4);
  task_data->inputs_count.emplace_back(4);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  agafeev_s_strassen_alg_seq::MultiplMatrixSequental testTask(task_data);
  bool isValid = testTask.ValidationImpl();
  ASSERT_EQ(isValid, false);
}

// TEST(agafeev_s_max_of_vector_elements_seq, find_max_in_10x10_matrix) {
//   const int rows = 10;
//   const int columns = 10;

//   // Create data
//   std::vector<int> in_matrix = create_RandomMatrix<int>(rows, columns);
//   std::vector<int> out(1, 0);
//   int right_answer = std::numeric_limits<int>::min();
//   for (auto &&t : in_matrix)
//     if (right_answer < t) right_answer = t;

//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
//   task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
//   task_data->inputs_count.emplace_back(in_matrix.size());
//   task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data->outputs_count.emplace_back(out.size());

//   // Create Task
//   agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int> testTask(task_data);
//   bool isValid = testTask.validation();
//   ASSERT_EQ(isValid, true);
//   testTask.pre_processing();
//   testTask.run();
//   testTask.post_processing();
//   ASSERT_EQ(right_answer, out[0]);
// }

// TEST(agafeev_s_max_of_vector_elements_seq, find_max_in_9x45_matrix) {
//   const int rows = 10;
//   const int columns = 10;

//   // Create data
//   std::vector<int> in_matrix = create_RandomMatrix<int>(rows, columns);
//   std::vector<int> out(1, 0);
//   int right_answer = std::numeric_limits<int>::min();
//   for (auto &&t : in_matrix)
//     if (right_answer < t) right_answer = t;

//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
//   task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
//   task_data->inputs_count.emplace_back(in_matrix.size());
//   task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data->outputs_count.emplace_back(out.size());

//   // Create Task
//   agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int> testTask(task_data);
//   bool isValid = testTask.validation();
//   ASSERT_EQ(isValid, true);
//   testTask.pre_processing();
//   testTask.run();
//   testTask.post_processing();
//   ASSERT_EQ(right_answer, out[0]);
// }

// TEST(agafeev_s_max_of_vector_elements_seq, find_max_in_130x187_matrix) {
//   const int rows = 10;
//   const int columns = 10;

//   // Create data
//   std::vector<int> in_matrix = create_RandomMatrix<int>(rows, columns);
//   std::vector<int> out(1, 0);
//   int right_answer = std::numeric_limits<int>::min();
//   for (auto &&t : in_matrix)
//     if (right_answer < t) right_answer = t;

//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
//   task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
//   task_data->inputs_count.emplace_back(in_matrix.size());
//   task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data->outputs_count.emplace_back(out.size());

//   // Create Task
//   agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int> testTask(task_data);
//   bool isValid = testTask.validation();
//   ASSERT_EQ(isValid, true);
//   testTask.pre_processing();
//   testTask.run();
//   testTask.post_processing();
//   ASSERT_EQ(right_answer, out[0]);
// }

// TEST(agafeev_s_max_of_vector_elements_seq, check_validate_func) {
//   // Create data
//   std::vector<int32_t> in(20, 1);
//   std::vector<int32_t> out(2, 0);

//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
//   task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data->inputs_count.emplace_back(in.size());
//   task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data->outputs_count.emplace_back(out.size());

//   // Create Task
//   agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int> testTask(task_data);
//   bool isValid = testTask.validation();
//   ASSERT_EQ(isValid, false);
// }

// TEST(agafeev_s_max_of_vector_elements_seq, check_wrong_order) {
//   // Create data
//   std::vector<float> in(20, 1);
//   std::vector<float> out(1, 0);

//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
//   task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data->inputs_count.emplace_back(in.size());
//   task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data->outputs_count.emplace_back(out.size());

//   // Create Task
//   agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int> testTask(task_data);
//   bool isValid = testTask.validation();
//   ASSERT_EQ(isValid, true);
//   testTask.pre_processing();
//   ASSERT_ANY_THROW(testTask.post_processing());
// }

// TEST(agafeev_s_max_of_vector_elements_seq, negative_numbers_test) {
//   // Create data
//   std::vector<int> in_matrix = {-20, -93, -93, -31, -56, -58, -16, -41, -88, -87, -35, -24, -4, -83, -54,
//                                 -93, -16, -44, -95, -87, -37, -15, -42, -82, -88, -18, -22, -2, -88, -94};
//   std::vector<int> out(1, 0);
//   int right_answer = std::numeric_limits<int>::min();
//   for (auto &&t : in_matrix)
//     if (right_answer < t) right_answer = t;

//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
//   task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
//   task_data->inputs_count.emplace_back(in_matrix.size());
//   task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data->outputs_count.emplace_back(out.size());

//   // Create Task
//   agafeev_s_max_of_vector_elements_seq::MaxMatrixSequental<int> testTask(task_data);
//   bool isValid = testTask.validation();
//   ASSERT_EQ(isValid, true);
//   testTask.pre_processing();
//   testTask.run();
//   testTask.post_processing();
//   ASSERT_EQ(right_answer, out[0]);
// }