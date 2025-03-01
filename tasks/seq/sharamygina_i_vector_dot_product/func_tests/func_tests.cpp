#include <gtest/gtest.h>

#include <random>

#include "core/task/include/task.hpp"
#include "seq/sharamygina_i_vector_dot_product/include/ops_seq.h"

namespace sharamygina_i_vector_dot_product_seq {
namespace {
int resulting(const std::vector<int>& v1, const std::vector<int>& v2) {
  int res = 0;
  for (size_t i = 0; i < v1.size(); ++i) {
    res += v1[i] * v2[i];
  }
  return res;
}
std::vector<int> GetVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> v(size);
  for (int i = 0; i < size; i++) {
    v[i] = gen() % 320 + gen() % 11;
  }
  return v;
}
}  // namespace
}  // namespace sharamygina_i_vector_dot_product_seq

TEST(sharamygina_i_vector_dot_product, SampleVecTest) {
  int size1 = 4;
  int size2 = 4;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(size1);
  taskData->inputs_count.emplace_back(size2);

  std::vector<int> received_res(1);
  int expected_res = 30;
  std::vector<int> v1 = {1, 2, 3, 4};
  std::vector<int> v2 = {1, 2, 3, 4};

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  taskData->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::vector_dot_product_seq testTask(taskData);
  ASSERT_TRUE(testTask.ValidationImpl());
  ASSERT_TRUE(testTask.PreProcessingImpl());
  ASSERT_TRUE(testTask.RunImpl());
  ASSERT_TRUE(testTask.PostProcessingImpl());

  ASSERT_EQ(received_res[0], expected_res);
}

TEST(sharamygina_i_vector_dot_product, BigVecTest) {
  int size1 = 200;
  int size2 = 200;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(size1);
  taskData->inputs_count.emplace_back(size2);

  std::vector<int> received_res(1);
  std::vector<int> v1(size1);
  std::vector<int> v2(size2);
  v1 = sharamygina_i_vector_dot_product_seq::GetVector(size1);
  v2 = sharamygina_i_vector_dot_product_seq::GetVector(size2);
  int expected_res = sharamygina_i_vector_dot_product_seq::resulting(v1, v2);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  taskData->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::vector_dot_product_seq testTask(taskData);
  ASSERT_TRUE(testTask.ValidationImpl());
  ASSERT_TRUE(testTask.PreProcessingImpl());
  ASSERT_TRUE(testTask.RunImpl());
  ASSERT_TRUE(testTask.PostProcessingImpl());

  ASSERT_EQ(received_res[0], expected_res);
}

TEST(sharamygina_i_vector_dot_product, DifferentVecValidationTest) {
  int size1 = 200;
  int size2 = 100;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(size1);
  taskData->inputs_count.emplace_back(size2);

  std::vector<int> received_res(1);
  std::vector<int> v1(size1);
  std::vector<int> v2(size2);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  taskData->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::vector_dot_product_seq testTask(taskData);
  ASSERT_FALSE(testTask.ValidationImpl());
}

TEST(sharamygina_i_vector_dot_product, OneSizeValidationTest) {
  int size1 = 200;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(size1);

  std::vector<int> received_res(1);
  std::vector<int> v1(size1);
  std::vector<int> v2(size1);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  taskData->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::vector_dot_product_seq testTask(taskData);
  ASSERT_FALSE(testTask.ValidationImpl());
}

TEST(sharamygina_i_vector_dot_product, OneVecValidationTest) {
  int size1 = 200;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(size1);
  taskData->inputs_count.emplace_back(size1);

  std::vector<int> received_res(1);
  std::vector<int> v1(size1);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  taskData->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::vector_dot_product_seq testTask(taskData);
  ASSERT_FALSE(testTask.ValidationImpl());
}

TEST(sharamygina_i_vector_dot_product, OneVecAndSizeValidationTest) {
  int size1 = 200;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(size1);

  std::vector<int> received_res(1);
  std::vector<int> v1(size1);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  taskData->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::vector_dot_product_seq testTask(taskData);
  ASSERT_FALSE(testTask.ValidationImpl());
}

TEST(sharamygina_i_vector_dot_product, MoreOutputCountValidationTest) {
  int size1 = 10;
  int size2 = 10;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(size1);
  taskData->inputs_count.emplace_back(size2);

  std::vector<int> received_res(1);
  std::vector<int> v1(size1);
  std::vector<int> v2(size2);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  taskData->outputs_count.emplace_back(received_res.size());
  taskData->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::vector_dot_product_seq testTask(taskData);
  ASSERT_FALSE(testTask.ValidationImpl());
}