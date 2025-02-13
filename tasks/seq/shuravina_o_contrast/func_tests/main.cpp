#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

TEST(shuravina_o_contrast, test_contrast_stretching_uniform_image) {
  constexpr size_t kSize = 256;

  std::vector<uint8_t> in(kSize * kSize, 128);
  std::vector<uint8_t> out(kSize * kSize, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskSequential contrast_task_sequential(task_data_seq);
  ASSERT_EQ(contrast_task_sequential.Validation(), true);
  contrast_task_sequential.PreProcessing();
  contrast_task_sequential.Run();
  contrast_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(out[i], 255);
  }
}

TEST(shuravina_o_contrast, test_contrast_stretching_varied_intensity) {
  constexpr size_t kSize = 256;

  std::vector<uint8_t> in(kSize * kSize);
  for (size_t i = 0; i < kSize * kSize; ++i) {
    in[i] = static_cast<uint8_t>(i % 256);
  }
  std::vector<uint8_t> out(kSize * kSize, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskSequential contrast_task_sequential(task_data_seq);
  ASSERT_EQ(contrast_task_sequential.Validation(), true);
  contrast_task_sequential.PreProcessing();
  contrast_task_sequential.Run();
  contrast_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(out[i], in[i]);
  }
}

TEST(shuravina_o_contrast, test_contrast_stretching_min_intensity) {
  constexpr size_t kSize = 256;

  std::vector<uint8_t> in(kSize * kSize, 0);
  std::vector<uint8_t> out(kSize * kSize, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskSequential contrast_task_sequential(task_data_seq);
  ASSERT_EQ(contrast_task_sequential.Validation(), true);
  contrast_task_sequential.PreProcessing();
  contrast_task_sequential.Run();
  contrast_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(out[i], 0);
  }
}

TEST(shuravina_o_contrast, test_contrast_stretching_max_intensity) {
  constexpr size_t kSize = 256;

  std::vector<uint8_t> in(kSize * kSize, 255);
  std::vector<uint8_t> out(kSize * kSize, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskSequential contrast_task_sequential(task_data_seq);
  ASSERT_EQ(contrast_task_sequential.Validation(), true);
  contrast_task_sequential.PreProcessing();
  contrast_task_sequential.Run();
  contrast_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(out[i], 255);
  }
}

TEST(shuravina_o_contrast, test_contrast_stretching_small_image) {
  constexpr size_t kSize = 16;

  std::vector<uint8_t> in(kSize * kSize);
  for (size_t i = 0; i < kSize * kSize; ++i) {
    in[i] = static_cast<uint8_t>(i % 256);
  }
  std::vector<uint8_t> out(kSize * kSize, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskSequential contrast_task_sequential(task_data_seq);
  ASSERT_EQ(contrast_task_sequential.Validation(), true);
  contrast_task_sequential.PreProcessing();
  contrast_task_sequential.Run();
  contrast_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(out[i], in[i]);
  }
}