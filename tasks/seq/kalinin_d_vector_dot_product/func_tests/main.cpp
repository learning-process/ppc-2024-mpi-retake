// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdint>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kalinin_d_vector_dot_product/include/ops_seq.hpp"

namespace {
int offset = 0;
}  // namespace

namespace {
std::vector<int> CreateRandomVector(int v_size) {
  std::vector<int> vec(v_size);
  std::mt19937 gen;
  gen.seed((unsigned)time(nullptr) + ++offset);
  for (int i = 0; i < v_size; i++) {
    vec[i] = static_cast<int>(gen() % 100);
  }
  return vec;
}
}  // namespace

TEST(kalinin_d_vector_dot_product_seq, can_scalar_multiply_vec_size_10) {
  const int count = 10;
  // Create data
  std::vector<int> out(1, 0);
  std::vector<int> v1 = CreateRandomVector(count);
  std::vector<int> v2 = CreateRandomVector(count);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  task_data_seq->inputs_count.emplace_back(v1.size());
  task_data_seq->inputs_count.emplace_back(v2.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_vector_dot_product_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  int answer = kalinin_d_vector_dot_product_seq::VectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}

TEST(kalinin_d_vector_dot_product_seq, can_scalar_multiply_vec_size_100) {
  const int count = 100;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = CreateRandomVector(count);
  std::vector<int> v2 = CreateRandomVector(count);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  task_data_seq->inputs_count.emplace_back(v1.size());
  task_data_seq->inputs_count.emplace_back(v2.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_vector_dot_product_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  int answer = kalinin_d_vector_dot_product_seq::VectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}

TEST(kalinin_d_vector_dot_product_seq, check_none_equal_size_of_vec) {
  const int count = 10;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = CreateRandomVector(count);
  std::vector<int> v2 = CreateRandomVector(count + 1);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  task_data_seq->inputs_count.emplace_back(v1.size());
  task_data_seq->inputs_count.emplace_back(v2.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_vector_dot_product_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(kalinin_d_vector_dot_product_seq, check_equal_size_of_vec) {
  const int count = 10;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = CreateRandomVector(count);
  std::vector<int> v2 = CreateRandomVector(count);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  task_data_seq->inputs_count.emplace_back(v1.size());
  task_data_seq->inputs_count.emplace_back(v2.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_vector_dot_product_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
}

TEST(kalinin_d_vector_dot_product_seq, check_empty_vec_product_func) {
  const int count = 0;
  std::vector<int> v1 = CreateRandomVector(count);
  std::vector<int> v2 = CreateRandomVector(count);
  int answer = kalinin_d_vector_dot_product_seq::VectorDotProduct(v1, v2);
  ASSERT_EQ(0, answer);
}

TEST(kalinin_d_vector_dot_product_seq, check_empty_vec_product_Run) {
  const int count = 0;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = CreateRandomVector(count);
  std::vector<int> v2 = CreateRandomVector(count);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  task_data_seq->inputs_count.emplace_back(v1.size());
  task_data_seq->inputs_count.emplace_back(v2.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_vector_dot_product_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  int answer = kalinin_d_vector_dot_product_seq::VectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}

TEST(kalinin_d_vector_dot_product_seq, v1_dot_product_v2_equal_v2_dot_product_v1) {
  const int count = 50;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = CreateRandomVector(count);
  std::vector<int> v2 = CreateRandomVector(count);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  task_data_seq->inputs_count.emplace_back(v1.size());
  task_data_seq->inputs_count.emplace_back(v2.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_vector_dot_product_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  int answer = kalinin_d_vector_dot_product_seq::VectorDotProduct(v2, v1);
  ASSERT_EQ(answer, out[0]);
}
TEST(kalinin_d_vector_dot_product_seq, check_Run_right) {
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = {1, 2, 5};
  std::vector<int> v2 = {4, 7, 8};

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  task_data_seq->inputs_count.emplace_back(v1.size());
  task_data_seq->inputs_count.emplace_back(v2.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_vector_dot_product_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_EQ(58, out[0]);
}
TEST(kalinin_d_vector_dot_product_seq, check_VectorDotProduct_right) {
  // Create data
  std::vector<int> v1 = {1, 2, 5};
  std::vector<int> v2 = {4, 7, 8};
  ASSERT_EQ(58, kalinin_d_vector_dot_product_seq::VectorDotProduct(v1, v2));
}

TEST(kalinin_d_vector_dot_product_seq, can_scalar_multiply_vec_size_20) {
  const int count = 20;
  // Create data
  std::vector<int> out(1, 0);
  std::vector<int> v1 = CreateRandomVector(count);
  std::vector<int> v2 = CreateRandomVector(count);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  task_data_seq->inputs_count.emplace_back(v1.size());
  task_data_seq->inputs_count.emplace_back(v2.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_vector_dot_product_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  int answer = kalinin_d_vector_dot_product_seq::VectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}

TEST(kalinin_d_vector_dot_product_seq, can_scalar_multiply_vec_size_50) {
  const int count = 50;
  // Create data
  std::vector<int> out(1, 0);
  std::vector<int> v1 = CreateRandomVector(count);
  std::vector<int> v2 = CreateRandomVector(count);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  task_data_seq->inputs_count.emplace_back(v1.size());
  task_data_seq->inputs_count.emplace_back(v2.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_vector_dot_product_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  int answer = kalinin_d_vector_dot_product_seq::VectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}

TEST(kalinin_d_vector_dot_product_seq, can_scalar_multiply_vec_size_200) {
  const int count = 200;
  // Create data
  std::vector<int> out(1, 0);
  std::vector<int> v1 = CreateRandomVector(count);
  std::vector<int> v2 = CreateRandomVector(count);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  task_data_seq->inputs_count.emplace_back(v1.size());
  task_data_seq->inputs_count.emplace_back(v2.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kalinin_d_vector_dot_product_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  int answer = kalinin_d_vector_dot_product_seq::VectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}