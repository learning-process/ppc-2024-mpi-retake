#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/Shpynov_N_radix_sort/include/Shpynov_N_radix_sort.hpp"

TEST(shpynov_n_radix_sort_seq, test_single_num) {
  std::vector<int> input_vec(1, 0);

  std::vector<int> expected_result(1, 0);
  std::vector<int> returned_result(1, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_seq->inputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_seq->outputs_count.emplace_back(1);

  shpynov_n_radix_sort_seq::TestTaskSEQ test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(expected_result, returned_result);
}

TEST(shpynov_n_radix_sort_seq, test_some_numbers) {
  std::vector<int> input_vec = {17, 33, 22, 42};

  std::vector<int> expected_result = {17, 22, 33, 42};
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_seq->inputs_count.emplace_back(input_vec.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_seq->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_seq::TestTaskSEQ test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(expected_result, returned_result);
}

TEST(shpynov_n_radix_sort_seq, test_some_numbers_diff_length) {
  std::vector<int> input_vec = {17, 33, 22, 420, 1, 0, 5837, 659};

  std::vector<int> expected_result = {0, 1, 17, 22, 33, 420, 659, 5837};
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_seq->inputs_count.emplace_back(input_vec.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_seq->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_seq::TestTaskSEQ test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(expected_result, returned_result);
}

TEST(shpynov_n_radix_sort_seq, test_some_numbers_diff_length_neg_numbers) {
  std::vector<int> input_vec = {-17, -33, -22, -420, -1, -5837, -659};

  std::vector<int> expected_result = {-5837, -659, -420, -33, -22, -17, -1};
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_seq->inputs_count.emplace_back(input_vec.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_seq->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_seq::TestTaskSEQ test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(expected_result, returned_result);
}

TEST(shpynov_n_radix_sort_seq, test_some_numbers_diff_length_pos_and_neg_numbers) {
  std::vector<int> input_vec = {17, 33, 22, 420, 1, 0, 5837, 659, -4, -28, -76, -110291};

  std::vector<int> expected_result = {-110291, -76, -28, -4, 0, 1, 17, 22, 33, 420, 659, 5837};
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_seq->inputs_count.emplace_back(input_vec.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_seq->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_seq::TestTaskSEQ test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(expected_result, returned_result);
}

TEST(shpynov_n_radix_sort_seq, test_some_numbers_diff_length_pos_and_neg_numbers_with_same_nums) {
  std::vector<int> input_vec = {17, 33, 22, 420, 1, 17, 0, 5837, 659, -4, -28, 0, -76, -4, -110291};

  std::vector<int> expected_result = {-110291, -76, -28, -4, -4, 0, 0, 1, 17, 17, 22, 33, 420, 659, 5837};
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_seq->inputs_count.emplace_back(input_vec.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_seq->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_seq::TestTaskSEQ test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(expected_result, returned_result);
}

TEST(shpynov_n_radix_sort_seq, test_invalid) {
  std::vector<int> input_vec;

  std::vector<int> expected_result = {-110291, -76, -28, -4, 0, 1, 17, 22, 33, 420, 659, 5837};
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_seq->inputs_count.emplace_back(input_vec.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_seq->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_seq::TestTaskSEQ test_task_seq(task_data_seq);
  ASSERT_NE(test_task_seq.ValidationImpl(), true);
}

TEST(shpynov_n_radix_sort_seq, test_random_number) {
  std::vector<int> input_vec = shpynov_n_radix_sort_seq::GetRandVec(1);
  std::vector<int> expected_result = input_vec;
  std::ranges::sort(expected_result.begin(), expected_result.end());
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_seq->inputs_count.emplace_back(input_vec.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_seq->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_seq::TestTaskSEQ test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(expected_result, returned_result);
}

TEST(shpynov_n_radix_sort_seq, test_random_small_vector) {
  std::vector<int> input_vec = shpynov_n_radix_sort_seq::GetRandVec(20);
  std::vector<int> expected_result = input_vec;
  std::ranges::sort(expected_result.begin(), expected_result.end());
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_seq->inputs_count.emplace_back(input_vec.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_seq->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_seq::TestTaskSEQ test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(expected_result, returned_result);
}

TEST(shpynov_n_radix_sort_seq, test_random_big_vector) {
  std::vector<int> input_vec = shpynov_n_radix_sort_seq::GetRandVec(1000);
  std::vector<int> expected_result = input_vec;
  std::ranges::sort(expected_result.begin(), expected_result.end());
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_seq->inputs_count.emplace_back(input_vec.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_seq->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_seq::TestTaskSEQ test_task_seq(task_data_seq);
  ASSERT_EQ(test_task_seq.ValidationImpl(), true);
  test_task_seq.PreProcessingImpl();
  test_task_seq.RunImpl();
  test_task_seq.PostProcessingImpl();

  ASSERT_EQ(expected_result, returned_result);
}