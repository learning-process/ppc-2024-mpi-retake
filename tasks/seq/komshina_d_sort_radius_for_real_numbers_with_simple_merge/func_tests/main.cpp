#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

using namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq;

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_sort_basic) {
  int size = 4;
  std::vector<double> in = {8.3, -4.7, 2.1, 3.5};
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs = {reinterpret_cast<uint8_t *>(&size), reinterpret_cast<uint8_t *>(in.data())};
  task_data_seq->inputs_count = {1, static_cast<unsigned int>(size)};
  task_data_seq->outputs = {reinterpret_cast<uint8_t *>(out.data())};
  task_data_seq->outputs_count = {static_cast<unsigned int>(size)};

  TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::ranges::sort(in);
  auto *result_seq = reinterpret_cast<double *>(task_data_seq->outputs[0]);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(in[i], result_seq[i], 1e-12);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_sort_negative_numbers) {
  int size = 4;
  std::vector<double> in = {-5.5, -1.1, -3.3, -2.2};
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs = {reinterpret_cast<uint8_t *>(&size), reinterpret_cast<uint8_t *>(in.data())};
  task_data_seq->inputs_count = {1, static_cast<unsigned int>(size)};
  task_data_seq->outputs = {reinterpret_cast<uint8_t *>(out.data())};
  task_data_seq->outputs_count = {static_cast<unsigned int>(size)};

  TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::ranges::sort(in);
  auto *result_seq = reinterpret_cast<double *>(task_data_seq->outputs[0]);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(in[i], result_seq[i], 1e-12);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_sort_large_numbers) {
  int size = 4;
  std::vector<double> in = {1e9, -1e9, 1e-9, -1e-9};
  std::vector<double> out(size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs = {reinterpret_cast<uint8_t *>(&size), reinterpret_cast<uint8_t *>(in.data())};
  task_data_seq->inputs_count = {1, static_cast<unsigned int>(size)};
  task_data_seq->outputs = {reinterpret_cast<uint8_t *>(out.data())};
  task_data_seq->outputs_count = {static_cast<unsigned int>(size)};

  TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::ranges::sort(in);
  auto *result_seq = reinterpret_cast<double *>(task_data_seq->outputs[0]);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(in[i], result_seq[i], 1e-12);
  }
}
