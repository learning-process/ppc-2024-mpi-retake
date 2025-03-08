#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/konkov_i_linear_hist_stretch_seq/include/ops_seq.hpp"

TEST(konkov_i_linear_hist_stretch_seq, test_basic_contrast) {
  constexpr size_t kSize = 256;

  std::vector<uint8_t> input(kSize);
  std::vector<uint8_t> output(kSize, 0);

  for (size_t i = 0; i < kSize; ++i) {
    input[i] = static_cast<uint8_t>(i);
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_seq->outputs_count.emplace_back(output.size());

  konkov_i_linear_hist_stretch_seq::LinearHistStretchSeq task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(output.front(), 0);
  EXPECT_EQ(output.back(), 255);
}

TEST(konkov_i_linear_hist_stretch_seq, test_constant_value) {
  constexpr size_t kSize = 256;
  std::vector<uint8_t> input(kSize, 100);
  std::vector<uint8_t> output(kSize, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(input.data());
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(output.data());
  task_data_seq->outputs_count.emplace_back(output.size());

  konkov_i_linear_hist_stretch_seq::LinearHistStretchSeq task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  for (auto val : output) {
    EXPECT_EQ(val, 100);
  }
}

TEST(konkov_i_linear_hist_stretch_seq, test_custom_range) {
  constexpr size_t kSize = 256;
  std::vector<uint8_t> input(kSize);
  input[0] = 50;
  input.back() = 200;
  for (size_t i = 1; i < input.size() - 1; ++i) {
    input[i] = 100;
  }

  std::vector<uint8_t> output(kSize, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(input.data());
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(output.data());
  task_data_seq->outputs_count.emplace_back(output.size());

  konkov_i_linear_hist_stretch_seq::LinearHistStretchSeq task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(output.front(), 0);
  EXPECT_EQ(output.back(), 255);
  EXPECT_EQ(output[1], 85);
}

TEST(konkov_i_linear_hist_stretch_seq, test_empty_image) {
  std::vector<uint8_t> in;
  std::vector<uint8_t> out;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konkov_i_linear_hist_stretch_seq::LinearHistStretchSeq task(task_data_seq);
  ASSERT_FALSE(task.Validation());
}