#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

TEST(shuravina_o_contrast, test_task_run) {
  constexpr size_t kSize = 256;
  std::vector<uint8_t> in(kSize * kSize, 0);
  std::vector<uint8_t> out(kSize * kSize, 0);

  for (size_t i = 0; i < kSize; ++i) {
    for (size_t j = 0; j < kSize; ++j) {
      in[i * kSize + j] = static_cast<uint8_t>(i + j);
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskSequential contrast_task_sequential(task_data_seq);

  ASSERT_TRUE(contrast_task_sequential.Validation());
  ASSERT_TRUE(contrast_task_sequential.PreProcessing());
  ASSERT_TRUE(contrast_task_sequential.Run());
  ASSERT_TRUE(contrast_task_sequential.PostProcessing());

  uint8_t max_val = *std::max_element(out.begin(), out.end());
  EXPECT_EQ(max_val, 255);
}