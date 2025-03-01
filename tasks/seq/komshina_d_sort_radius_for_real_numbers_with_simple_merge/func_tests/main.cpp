#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

using namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq;

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, AlreadySortedAscending) {
  int size = 5;
  std::vector<double> in = {-3.5, -2.1, 0.0, 1.1, 5.6};
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(size);

  TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);

  for (int i = 0; i < size; ++i) {
    double diff = std::fabs(in[i] - result_seq[i]);
    EXPECT_LT(diff, 1e-12) << "Values at index " << i << " differ by more than the allowed tolerance.";
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, AlreadySortedDescending) {
  int size = 5;
  std::vector<double> in = {5.6, 1.1, 0.0, -2.1, -3.5};
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(size);

  TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);
  std::ranges::sort(in);

  for (int i = 0; i < size; ++i) {
    double diff = std::fabs(in[i] - result_seq[i]);
    EXPECT_LT(diff, 1e-12) << "Values at index " << i << " differ by more than the allowed tolerance.";
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, AllEqualNumbers) {
  int size = 7;
  std::vector<double> in(size, 3.14);
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(size);

  TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);

  for (int i = 0; i < size; ++i) {
    double diff = std::fabs(in[i] - result_seq[i]);
    EXPECT_LT(diff, 1e-12) << "Values at index " << i << " differ by more than the allowed tolerance.";
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, MixedPositiveAndNegative) {
  int size = 6;
  std::vector<double> in = {4.2, -1.5, 3.3, -0.5, -6.1, 2.8};
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(size);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(size);

  TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  auto* result_seq = reinterpret_cast<double*>(task_data_seq->outputs[0]);
  std::ranges::sort(in);

  for (int i = 0; i < size; ++i) {
    double diff = std::fabs(in[i] - result_seq[i]);
    EXPECT_LT(diff, 1e-12) << "Values at index " << i << " differ by more than the allowed tolerance.";
  }
}