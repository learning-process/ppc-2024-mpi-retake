// Anikin Maksim 2025
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/anikin_m_counting_characters/include/ops_seq.hpp"

TEST(anikin_m_counting_characters_seq, one_char_dif) {
  // Create data
  std::vector<char> in1;
  anikin_m_counting_characters_seq::CreateDataVector(&in1, "aboba");
  std::vector<char> in2;
  anikin_m_counting_characters_seq::CreateDataVector(&in2, "ababa");
  int res_out = 0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_out));
  task_data_seq->outputs_count.emplace_back(1);
  // Create Task
  anikin_m_counting_characters_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(res_out, 1);
}

TEST(anikin_m_counting_characters_seq, first_larger) {
  // Create data
  std::vector<char> in1;
  anikin_m_counting_characters_seq::CreateDataVector(&in1, "abobaa");
  std::vector<char> in2;
  anikin_m_counting_characters_seq::CreateDataVector(&in2, "ababa");
  int res_out = 0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_out));
  task_data_seq->outputs_count.emplace_back(1);
  // Create Task
  anikin_m_counting_characters_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(res_out, 2);
}

TEST(anikin_m_counting_characters_seq, second_larger) {
  // Create data
  std::vector<char> in1;
  anikin_m_counting_characters_seq::CreateDataVector(&in1, "aboba");
  std::vector<char> in2;
  anikin_m_counting_characters_seq::CreateDataVector(&in2, "ababaa");
  int res_out = 0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_seq->inputs_count.emplace_back(in1.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_seq->inputs_count.emplace_back(in2.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_out));
  task_data_seq->outputs_count.emplace_back(1);
  // Create Task
  anikin_m_counting_characters_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(res_out, 2);
}