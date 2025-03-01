#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sharamygina_i_vector_dot_product/include/ops_seq.h"

namespace sharamygina_i_vector_dot_product_seq {
namespace {
int Resulting(const std::vector<int>& v1, const std::vector<int>& v2) {
  int res = 0;
  for (unsigned int i = 0; i < v1.size(); ++i) {
    res += v1[i] * v2[i];
  }
  return res;
}
std::vector<int> GetVector(unsigned int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> v(size);
  for (unsigned int i = 0; i < size; i++) {
    v[i] = gen() % 320 + gen() % 11;
  }
  return v;
}
}  // namespace
}  // namespace sharamygina_i_vector_dot_product_seq

TEST(sharamygina_i_vector_dot_product, SampleVecTest) {
  unsigned int kSize1 = 4;
  unsigned int kSize2 = 4;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(kSize1);
  task_data->inputs_count.emplace_back(kSize2);

  std::vector<int> received_res(1);
  int expected_res = 30;
  std::vector<int> v1 = {1, 2, 3, 4};
  std::vector<int> v2 = {1, 2, 3, 4};

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  task_data->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::VectorDotProductSeq test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  ASSERT_EQ(received_res[0], expected_res);
}

TEST(sharamygina_i_vector_dot_product, BigVecTest) {
  unsigned int kSize1 = 200;
  unsigned int kSize2 = 200;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(kSize1);
  task_data->inputs_count.emplace_back(kSize2);

  std::vector<int> received_res(1);
  std::vector<int> v1(kSize1);
  std::vector<int> v2(kSize2);
  v1 = sharamygina_i_vector_dot_product_seq::GetVector(kSize1);
  v2 = sharamygina_i_vector_dot_product_seq::GetVector(kSize2);
  int expected_res = sharamygina_i_vector_dot_product_seq::Resulting(v1, v2);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  task_data->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::VectorDotProductSeq test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  ASSERT_EQ(received_res[0], expected_res);
}

TEST(sharamygina_i_vector_dot_product, DifferentVecValidationTest) {
  unsigned int kSize1 = 200;
  unsigned int kSize2 = 100;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(kSize1);
  task_data->inputs_count.emplace_back(kSize2);

  std::vector<int> received_res(1);
  std::vector<int> v1(kSize1);
  std::vector<int> v2(kSize2);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  task_data->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::VectorDotProductSeq test_task(task_data);
  ASSERT_FALSE(test_task.ValidationImpl());
}

TEST(sharamygina_i_vector_dot_product, OneSizeValidationTest) {
  unsigned int kSize1 = 200;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(kSize1);

  std::vector<int> received_res(1);
  std::vector<int> v1(kSize1);
  std::vector<int> v2(kSize1);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  task_data->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::VectorDotProductSeq test_task(task_data);
  ASSERT_FALSE(test_task.ValidationImpl());
}

TEST(sharamygina_i_vector_dot_product, OneVecValidationTest) {
  unsigned int kSize1 = 200;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(kSize1);
  task_data->inputs_count.emplace_back(kSize1);

  std::vector<int> received_res(1);
  std::vector<int> v1(kSize1);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  task_data->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::VectorDotProductSeq test_task(task_data);
  ASSERT_FALSE(test_task.ValidationImpl());
}

TEST(sharamygina_i_vector_dot_product, OneVecAndSizeValidationTest) {
  unsigned int kSize1 = 200;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(kSize1);

  std::vector<int> received_res(1);
  std::vector<int> v1(kSize1);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  task_data->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::VectorDotProductSeq test_task(task_data);
  ASSERT_FALSE(test_task.ValidationImpl());
}

TEST(sharamygina_i_vector_dot_product, MoreOutputCountValidationTest) {
  unsigned int kSize1 = 10;
  unsigned int kSize2 = 10;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(kSize1);
  task_data->inputs_count.emplace_back(kSize2);

  std::vector<int> received_res(1);
  std::vector<int> v1(kSize1);
  std::vector<int> v2(kSize2);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  task_data->outputs_count.emplace_back(received_res.size());
  task_data->outputs_count.emplace_back(received_res.size());

  sharamygina_i_vector_dot_product_seq::VectorDotProductSeq test_task(task_data);
  ASSERT_FALSE(test_task.ValidationImpl());
}