#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/makadrai_a_sobel/include/ops_seq.hpp"

TEST(makadrai_a_sobel_seq, test_2_2) {
  int height_img = 2;
  int width_img = 2;
  std::vector<int> img = {100, 50, 150, 200};
  std::vector<int> res(4);

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

  task_data->inputs_count.emplace_back(width_img);
  task_data->inputs_count.emplace_back(height_img);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(width_img);
  task_data->outputs_count.emplace_back(height_img);

  std::vector<int> ans = {228, 255, 201, 175};

  makadrai_a_sobel_seq::Sobel sobel(task_data);
  ASSERT_TRUE(sobel.Validation());
  sobel.PreProcessing();
  sobel.Run();
  sobel.PostProcessing();
  EXPECT_EQ(ans, res);
}

TEST(makadrai_a_sobel_seq, test_1_1) {
  int height_img = 1;
  int width_img = 1;
  std::vector<int> img = {100};
  std::vector<int> res(1);

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

  task_data->inputs_count.emplace_back(width_img);
  task_data->inputs_count.emplace_back(height_img);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(width_img);
  task_data->outputs_count.emplace_back(height_img);

  std::vector<int> ans = {0};

  makadrai_a_sobel_seq::Sobel sobel(task_data);
  ASSERT_TRUE(sobel.Validation());
  sobel.PreProcessing();
  sobel.Run();
  sobel.PostProcessing();
  EXPECT_EQ(ans, res);
}

TEST(makadrai_a_sobel_seq, test_3_3) {
  int height_img = 3;
  int width_img = 3;
  std::vector<int> img = {32, 21, 61, 201, 231, 61, 132, 61, 111};
  std::vector<int> res(9);

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

  task_data->inputs_count.emplace_back(width_img);
  task_data->inputs_count.emplace_back(height_img);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(width_img);
  task_data->outputs_count.emplace_back(height_img);

  std::vector<int> ans = {235, 248, 152, 203, 121, 191, 247, 255, 170};

  makadrai_a_sobel_seq::Sobel sobel(task_data);
  ASSERT_TRUE(sobel.Validation());
  sobel.PreProcessing();
  sobel.Run();
  sobel.PostProcessing();
  EXPECT_EQ(ans, res);
}

TEST(makadrai_a_sobel_seq, test_error_1) {
  int height_img = 3;
  int width_img = 3;
  std::vector<int> img = {32, 21, 61, 201, 231, 61, 132, 61, 111};
  std::vector<int> res(9);

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

  task_data->inputs_count.emplace_back(width_img);
  task_data->inputs_count.emplace_back(height_img);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(width_img + 1);
  task_data->outputs_count.emplace_back(height_img);

  std::vector<int> ans = {235, 248, 152, 203, 121, 191, 247, 255, 170};

  makadrai_a_sobel_seq::Sobel sobel(task_data);
  ASSERT_FALSE(sobel.Validation());
}

TEST(makadrai_a_sobel_seq, test_error_2) {
  int height_img = 3;
  int width_img = 3;
  std::vector<int> img = {32, 21, 61, 201, 231, 61, 132, 61, 111};
  std::vector<int> res(9);

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

  task_data->inputs_count.emplace_back(width_img + 1);
  task_data->inputs_count.emplace_back(height_img);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(width_img);
  task_data->outputs_count.emplace_back(height_img);

  std::vector<int> ans = {235, 248, 152, 203, 121, 191, 247, 255, 170};

  makadrai_a_sobel_seq::Sobel sobel(task_data);
  ASSERT_FALSE(sobel.Validation());
}

TEST(makadrai_a_sobel_seq, test_error_3) {
  int height_img = 3;
  int width_img = 3;
  std::vector<int> img = {32, 21, 61, 201, 231, 61, 132, 61, 111};
  std::vector<int> res(8);

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

  task_data->inputs_count.emplace_back(width_img + 1);
  task_data->inputs_count.emplace_back(height_img);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(width_img);
  task_data->outputs_count.emplace_back(height_img);

  std::vector<int> ans = {235, 248, 152, 203, 121, 191, 247, 255, 170};

  makadrai_a_sobel_seq::Sobel sobel(task_data);
  ASSERT_FALSE(sobel.Validation());
}