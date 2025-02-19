#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/karaseva_e_num_of_alternations_signs/include/ops_seq.hpp"

TEST(karaseva_e_num_of_alternations_signs_seq, test_alternation) {
  constexpr size_t kCount = 50;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(1, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = (i % 2 == 0) ? 1 : -1;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  karaseva_e_num_of_alternations_signs_seq::AlternatingSignsSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(static_cast<long>(out[0]), static_cast<long>(kCount - 1));
}

TEST(karaseva_e_num_of_alternations_signs_seq, test_alternations_every_two) {
  constexpr size_t kCount = 50;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(1, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = (i / 2 % 2 == 0) ? 1 : -1;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  karaseva_e_num_of_alternations_signs_seq::AlternatingSignsSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(static_cast<long>(out[0]), static_cast<long>((kCount / 2) - 1));
}

TEST(karaseva_e_num_of_alternations_signs_seq, test_no_alternations) {
  constexpr size_t kCount = 50;

  std::vector<int> in(kCount, 1);
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  karaseva_e_num_of_alternations_signs_seq::AlternatingSignsSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(static_cast<long>(out[0]), static_cast<long>(0));
}
