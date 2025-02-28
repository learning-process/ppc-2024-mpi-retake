
#include <gtest/gtest.h>

#include <limits>
#include <random>
#include <vector>
#include <memory> 
#include <utility>
#include <stdint.h>
#include <algorithm>

#include "seq/vinyaikina_e_max_of_vector_elements/include/ops_seq.hpp"

TEST(vinyaikina_e_max_of_vector_elements, regularVector) {
  std::ranges::vector<int32_t> input = {1, 2, 3, -5, 3, 43};
  int32_t expected = 43;
  int32_t actual = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(input.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->outputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  vinyaikina_e_max_of_vector_elements_seq::VectorMaxSeq vector_max_seq(task_data);
  ASSERT_TRUE(vector_max_seq.ValidationImpl());
  vector_max_seq.PreProcessingImpl();
  vector_max_seq.RunImpl();
  vector_max_seq.PostProcessingImpl();
  ASSERT_EQ(expected, actual);
}

TEST(vinyaikina_e_max_of_vector_elements, positiveNumbers) {
  std::vector<int32_t> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int32_t expected = 10;
  int32_t actual = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(input.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->outputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  vinyaikina_e_max_of_vector_elements_seq::VectorMaxSeq vector_max_seq(task_data);
  ASSERT_TRUE(vector_max_seq.ValidationImpl());
  vector_max_seq.PreProcessingImpl();
  vector_max_seq.RunImpl();
  vector_max_seq.PostProcessingImpl();
  ASSERT_EQ(expected, actual);
}

TEST(vinyaikina_e_max_of_vector_elements, negativeNumbers) {
  std::vector<int32_t> input = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
  int32_t expected = -1;
  int32_t actual = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(input.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->outputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  vinyaikina_e_max_of_vector_elements_seq::VectorMaxSeq vector_max_seq(task_data);
  ASSERT_TRUE(vector_max_seq.ValidationImpl());
  vector_max_seq.PreProcessingImpl();
  vector_max_seq.RunImpl();
  vector_max_seq.PostProcessingImpl();
  ASSERT_EQ(expected, actual);
}

TEST(vinyaikina_e_max_of_vector_elements, zeroVector) {
  std::vector<int32_t> input = {0, 0, 0, 0};
  int32_t expected = 0;
  int32_t actual = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(input.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->outputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  vinyaikina_e_max_of_vector_elements_seq::VectorMaxSeq vector_max_seq(task_data);
  ASSERT_TRUE(vector_max_seq.ValidationImpl());
  vector_max_seq.PreProcessingImpl();
  vector_max_seq.RunImpl();
  vector_max_seq.PostProcessingImpl();
  ASSERT_EQ(expected, actual);
}

TEST(vinyaikina_e_max_of_vector_elements, randomVector) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-1000, 1000);

  std::vector<int32_t> input_vector(50000);
  std::generate(input_vector.begin(), input_vector.end(), [&]() { return distrib(gen); });

  int32_t expected_max = *std::ranges::max_element(input_vector.begin(), input_vector.end());

  int32_t actual_max = std::numeric_limits<int32_t>::min();
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
  task_data->outputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual_max));

  vinyaikina_e_max_of_vector_elements_seq::VectorMaxSeq vector_max_seq(task_data);
  ASSERT_TRUE(vector_max_seq.ValidationImpl());
  vector_max_seq.PreProcessingImpl();
  vector_max_seq.RunImpl();
  vector_max_seq.PostProcessingImpl();

  ASSERT_EQ(expected_max, actual_max);
}

TEST(vinyaikina_e_max_of_vector_elements, emptyVector) {
  std::vector<int32_t> input = {};
  int32_t actual = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(input.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->outputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  vinyaikina_e_max_of_vector_elements_seq::VectorMaxSeq vector_max_seq(task_data);
  ASSERT_TRUE(vector_max_seq.ValidationImpl());
  vector_max_seq.PreProcessingImpl();
  vector_max_seq.RunImpl();
  vector_max_seq.PostProcessingImpl();
  ASSERT_EQ(std::numeric_limits<int32_t>::min(), actual);
}

TEST(vinyaikina_e_max_of_vector_elements, validationNotPassed) {
  std::vector<int32_t> input = {1, 2, 3, -5};

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.emplace_back(input.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));

  vinyaikina_e_max_of_vector_elements_seq::VectorMaxSeq vector_max_seq(task_data);
  ASSERT_FALSE(vector_max_seq.ValidationImpl());
}
