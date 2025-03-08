#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "core/task/include/task.hpp"
#include "seq/dudchenko_o_sleeping_barber/include/ops_seq.hpp"

TEST(dudchenko_o_sleeping_barber_sequential, validation_test_1) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  int output_value = 0;
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_value));
  task_data_seq->outputs_count.emplace_back(sizeof(int));

  dudchenko_o_sleeping_barber_seq::TestSleepingBarber test_sleeping_barber(task_data_seq);
  task_data_seq->inputs_count = {0};
  EXPECT_FALSE(test_sleeping_barber.ValidationImpl());
}

TEST(dudchenko_o_sleeping_barber_sequential, validation_test_2) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  int output_value = 0;
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_value));
  task_data_seq->outputs_count.emplace_back(sizeof(int));

  dudchenko_o_sleeping_barber_seq::TestSleepingBarber test_sleeping_barber(task_data_seq);
  task_data_seq->inputs_count = {1};
  EXPECT_FALSE(test_sleeping_barber.ValidationImpl());
}

TEST(dudchenko_o_sleeping_barber_sequential, validation_test_3) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  int output_value = 0;
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_value));
  task_data_seq->outputs_count.emplace_back(sizeof(int));

  dudchenko_o_sleeping_barber_seq::TestSleepingBarber test_sleeping_barber(task_data_seq);
  task_data_seq->inputs_count = {5};
  EXPECT_TRUE(test_sleeping_barber.ValidationImpl());
}

TEST(dudchenko_o_sleeping_barber_seq, functional_test_small) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  int global_res = -1;

  task_data_seq->inputs_count.emplace_back(3);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  task_data_seq->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_seq::TestSleepingBarber test_sleeping_barber(task_data_seq);

  ASSERT_TRUE(test_sleeping_barber.ValidationImpl());
  ASSERT_TRUE(test_sleeping_barber.PreProcessingImpl());
  ASSERT_TRUE(test_sleeping_barber.RunImpl());
  ASSERT_TRUE(test_sleeping_barber.PostProcessingImpl());

  EXPECT_EQ(global_res, 0);
}

TEST(dudchenko_o_sleeping_barber_seq, functional_test_large) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  int global_res = -1;

  task_data_seq->inputs_count.emplace_back(1024);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  task_data_seq->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_seq::TestSleepingBarber test_sleeping_barber(task_data_seq);

  ASSERT_TRUE(test_sleeping_barber.ValidationImpl());
  ASSERT_TRUE(test_sleeping_barber.PreProcessingImpl());
  ASSERT_TRUE(test_sleeping_barber.RunImpl());
  ASSERT_TRUE(test_sleeping_barber.PostProcessingImpl());

  EXPECT_EQ(global_res, 0);
}
