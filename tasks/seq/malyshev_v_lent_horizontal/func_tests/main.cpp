#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"
#include "seq/malyshev_v_lent_horizontal/include/ops_seq.hpp"

namespace malyshev_v_lent_horizontal_seq {
namespace {
std::vector<double> GetRandomMatrix(size_t rows, size_t cols) {
  std::vector<double> matrix(rows * cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      matrix[i * cols + j] = static_cast<double>(rand()) / RAND_MAX * 100.0;
    }
  }
  return matrix;
}

std::vector<double> GetRandomVector(size_t size) {
  std::vector<double> vector(size);
  for (size_t i = 0; i < size; ++i) {
    vector[i] = static_cast<double>(rand()) / RAND_MAX * 100.0;
  }
  return vector;
}
}  // namespace
}  // namespace malyshev_v_lent_horizontal_seq

TEST(malyshev_v_lent_horizontal_seq, Validation_Test) {
  const size_t rows = 2;
  const size_t cols = 3;

  std::vector<double> matrix = {1, 2, 3, 4, 5, 6};
  std::vector<double> vector = {1, 2, 3};
  std::vector<double> result(rows, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(cols);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(rows);

  malyshev_v_lent_horizontal_seq::MatrixVectorMultiplication task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
}

TEST(malyshev_v_lent_horizontal_seq, Simple_Test) {
  const size_t rows = 2;
  const size_t cols = 3;

  std::vector<double> matrix = {1, 2, 3, 4, 5, 6};
  std::vector<double> vector = {1, 2, 3};
  std::vector<double> result(rows, 0.0);
  std::vector<double> expected = {14, 32};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(cols);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(rows);

  malyshev_v_lent_horizontal_seq::MatrixVectorMultiplication task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  ASSERT_EQ(result, expected);
}

TEST(malyshev_v_lent_horizontal_seq, Random_Test) {
  const size_t rows = 100;
  const size_t cols = 100;

  auto matrix = malyshev_v_lent_horizontal_seq::GetRandomMatrix(rows, cols);
  auto vector = malyshev_v_lent_horizontal_seq::GetRandomVector(cols);
  std::vector<double> result(rows, 0.0);

  std::vector<double> expected(rows, 0.0);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      expected[i] += matrix[i * cols + j] * vector[j];
    }
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(cols);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(rows);

  malyshev_v_lent_horizontal_seq::MatrixVectorMultiplication task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  ASSERT_EQ(result, expected);
}