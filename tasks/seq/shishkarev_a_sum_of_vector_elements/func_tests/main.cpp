// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shishkarev_a_sum_of_vector_elements/include/ops_seq.hpp"

TEST(shishkarev_a_sum_of_vector_elements_seq, test_int) {
  std::vector<int32_t> input_data(1, 10);
  const int32_t expected_sum = 10;
  std::vector<int32_t> output_data(1, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int32_t> task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  ASSERT_EQ(output_data[0], expected_sum);
}

TEST(shishkarev_a_sum_of_vector_elements_seq, test_float) {
  std::vector<float> input_data(1, 1.0F);
  std::vector<float> output_data(1, 0.0F);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<float> task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  EXPECT_NEAR(output_data[0], 1.0F, 1e-3F);
}

TEST(shishkarev_a_sum_of_vector_elements_seq, test_double) {
  std::vector<double> input_data(1, 10.0);
  const double expected_sum = 10.0;
  std::vector<double> output_data(1, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<double> task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  EXPECT_NEAR(output_data[0], expected_sum, 1e-6);
}

TEST(shishkarev_a_sum_of_vector_elements_seq, test_int64_t) {
  std::vector<int64_t> input_data(75836, 1);
  std::vector<int64_t> output_data(1, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int64_t> task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  ASSERT_EQ(output_data[0], static_cast<int64_t>(input_data.size()));
}

TEST(shishkarev_a_sum_of_vector_elements_seq, test_uint8_t) {
  std::vector<uint8_t> input_data(255, 1);
  std::vector<uint8_t> output_data(1, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<uint8_t> task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  ASSERT_EQ(output_data[0], static_cast<uint8_t>(input_data.size()));
}

TEST(shishkarev_a_sum_of_vector_elements_seq, test_empty) {
  std::vector<int32_t> input_data(1, 0);
  const int32_t expected_sum = 0;
  std::vector<int32_t> output_data(1, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.emplace_back(output_data.size());

  shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int32_t> task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  ASSERT_EQ(output_data[0], expected_sum);
}
