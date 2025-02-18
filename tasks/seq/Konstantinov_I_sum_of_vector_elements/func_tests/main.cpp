#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/Konstantinov_I_sum_of_vector_elements/include/ops_seq.hpp"

std::vector<int> generate_rand_vector(int size, int lower_bound = 0, int upper_bound = 50) {
  std::vector<int> result(size);
  for (int i = 0; i < size; i++) {
    result[i] = lower_bound + rand() % (upper_bound - lower_bound + 1);
  }
  return result;
}

std::vector<std::vector<int>> generate_rand_matrix(int rows, int columns, int lower_bound = 0, int upper_bound = 50) {
  std::vector<std::vector<int>> result(rows);
  for (int i = 0; i < rows; i++) {
    result[i] = generate_rand_vector(columns, lower_bound, upper_bound);
  }
  return result;
  return std::vector<std::vector<int>>();
}

TEST(Konstantinov_I_sum_seq, EmptyInput) {
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  Konstantinov_I_sum_of_vector_elements_seq::SumVecElemSequential test(taskDataPar);
  ASSERT_FALSE(test.ValidationImpl());
}

TEST(Konstantinov_I_sum_of_vector_seq, EmptyOutput) {
  int rows = 10;
  int columns = 10;
  std::vector<std::vector<int>> input = generate_rand_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }

  Konstantinov_I_sum_of_vector_elements_seq::SumVecElemSequential test(taskDataPar);
  ASSERT_FALSE(test.ValidationImpl());
}

TEST(Konstantinov_I_sum_of_vector_seq, EmptyMatrix) {
  int rows = 0;
  int columns = 0;
  int result;
  std::vector<std::vector<int>> input = generate_rand_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  Konstantinov_I_sum_of_vector_elements_seq::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.ValidationImpl());
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();
  ASSERT_EQ(0, result);
}

TEST(Konstantinov_I_sum_of_vector_seq, Matrix1x1) {
  int rows = 1;
  int columns = 1;
  int result;
  std::vector<std::vector<int>> input = generate_rand_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  Konstantinov_I_sum_of_vector_elements_seq::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.ValidationImpl());
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();
  ASSERT_EQ(sum, result);
}

TEST(Konstantinov_I_sum_of_vector_seq, Matrix5x1) {
  int rows = 5;
  int columns = 1;
  int result;
  std::vector<std::vector<int>> input = generate_rand_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  Konstantinov_I_sum_of_vector_elements_seq::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.ValidationImpl());
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();
  ASSERT_EQ(sum, result);
}

TEST(Konstantinov_I_sum_of_vector_seq, Matrix10x10) {
  int rows = 10;
  int columns = 10;
  int result;
  std::vector<std::vector<int>> input = generate_rand_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  Konstantinov_I_sum_of_vector_elements_seq::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.ValidationImpl());
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();
  ASSERT_EQ(sum, result);
}

TEST(Konstantinov_I_sum_of_vector_seq, Matrix100x100) {
  int rows = 100;
  int columns = 100;
  int result;
  std::vector<std::vector<int>> input = generate_rand_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  Konstantinov_I_sum_of_vector_elements_seq::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.ValidationImpl());
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();
  ASSERT_EQ(sum, result);
}

TEST(Konstantinov_I_sum_of_vector_seq, Matrix100x10) {
  int rows = 100;
  int columns = 10;
  int result;
  std::vector<std::vector<int>> input = generate_rand_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  Konstantinov_I_sum_of_vector_elements_seq::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.ValidationImpl());
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();
  ASSERT_EQ(sum, result);
}

TEST(Konstantinov_I_sum_of_vector_seq, Matrix10x100) {
  int rows = 10;
  int columns = 100;
  int result;
  std::vector<std::vector<int>> input = generate_rand_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  Konstantinov_I_sum_of_vector_elements_seq::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.ValidationImpl());
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();
  ASSERT_EQ(sum, result);
}