#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/khovansky_d_num_of_alternations_signs/include/ops_seq.hpp"

TEST(khovansky_d_num_of_alternations_signs_seq, test_10) {
  // Create data
  std::vector<int> in = {1, 2, -3, -4, -5, 6, -7, 8, 9, 10};
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_num_of_alternations_signs_seq::NumOfAlternationsSignsSeq numOfAlternationsSignsSeq(task_data_seq);
  ASSERT_EQ(numOfAlternationsSignsSeq.ValidationImpl(), true);
  numOfAlternationsSignsSeq.PreProcessingImpl();
  numOfAlternationsSignsSeq.RunImpl();
  numOfAlternationsSignsSeq.PostProcessingImpl();
  ASSERT_EQ(4, out[0]);
}

TEST(khovansky_d_num_of_alternations_signs_seq, invalid_input) {
  // Create data
  std::vector<int> in = {1};
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_num_of_alternations_signs_seq::NumOfAlternationsSignsSeq numOfAlternationsSignsSeq(task_data_seq);
  ASSERT_EQ(numOfAlternationsSignsSeq.ValidationImpl(), false);
}

TEST(khovansky_d_num_of_alternations_signs_seq, test_with_zero) {
  // Create data
  std::vector<int> in = {1, 0, -1, 0, 0, -1, -1, 1};
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_num_of_alternations_signs_seq::NumOfAlternationsSignsSeq numOfAlternationsSignsSeq(task_data_seq);
  ASSERT_EQ(numOfAlternationsSignsSeq.ValidationImpl(), true);
  numOfAlternationsSignsSeq.PreProcessingImpl();
  numOfAlternationsSignsSeq.RunImpl();
  numOfAlternationsSignsSeq.PostProcessingImpl();
  ASSERT_EQ(4, out[0]);
}

TEST(khovansky_d_num_of_alternations_signs_seq, test_with_only_zero) {
  // Create data
  std::vector<int> in = {0, 0, 0, 0, 0, 0};
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_num_of_alternations_signs_seq::NumOfAlternationsSignsSeq numOfAlternationsSignsSeq(task_data_seq);
  ASSERT_EQ(numOfAlternationsSignsSeq.ValidationImpl(), true);
  numOfAlternationsSignsSeq.PreProcessingImpl();
  numOfAlternationsSignsSeq.RunImpl();
  numOfAlternationsSignsSeq.PostProcessingImpl();
  ASSERT_EQ(0, out[0]);
}

TEST(khovansky_d_num_of_alternations_signs_seq, test_with_only_positive) {
  // Create data
  std::vector<int> in = {1, 0, 1, 0, 0, 1, 1, 1};
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_num_of_alternations_signs_seq::NumOfAlternationsSignsSeq numOfAlternationsSignsSeq(task_data_seq);
  ASSERT_EQ(numOfAlternationsSignsSeq.ValidationImpl(), true);
  numOfAlternationsSignsSeq.PreProcessingImpl();
  numOfAlternationsSignsSeq.RunImpl();
  numOfAlternationsSignsSeq.PostProcessingImpl();
  ASSERT_EQ(0, out[0]);
}

TEST(khovansky_d_num_of_alternations_signs_seq, test_with_only_negative) {
  // Create data
  std::vector<int> in = {-1, -1, -1};
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_num_of_alternations_signs_seq::NumOfAlternationsSignsSeq numOfAlternationsSignsSeq(task_data_seq);
  ASSERT_EQ(numOfAlternationsSignsSeq.ValidationImpl(), true);
  numOfAlternationsSignsSeq.PreProcessingImpl();
  numOfAlternationsSignsSeq.RunImpl();
  numOfAlternationsSignsSeq.PostProcessingImpl();
  ASSERT_EQ(0, out[0]);
}