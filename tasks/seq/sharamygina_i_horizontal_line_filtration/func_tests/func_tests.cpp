#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sharamygina_i_horizontal_line_filtration/include/ops_seq.h"

namespace sharamygina_i_horizontal_line_filtration_seq {
namespace {
void ToFiltSeq(const std::vector<unsigned int>& input, int rows, int cols, std::vector<unsigned int>& output) {
  const int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  output.assign(rows * cols, 0);

  for (int r = 1; r < rows - 1; ++r) {
    for (int c = 1; c < cols - 1; ++c) {
      unsigned int sum = 0;
      for (int kr = -1; kr <= 1; ++kr) {
        for (int kc = -1; kc <= 1; ++kc) {
          sum += input[((r + kr) * cols) + (c + kc)] * kernel[kr + 1][kc + 1];
        }
      }
      output[(r * cols) + c] = sum / 16;
    }
  }
}

std::vector<unsigned int> GetImage(int rows, int cols) {
  std::vector<unsigned int> temporary_im(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> dist(0, std::numeric_limits<unsigned int>::max());
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      temporary_im[(i * cols) + j] = dist(gen);
    }
  }
  return temporary_im;
}
}  // namespace
}  // namespace sharamygina_i_horizontal_line_filtration_seq

TEST(sharamygina_i_horizontal_line_filtration, SampleImageTest) {
  int rows = 4;
  int cols = 4;

  std::shared_ptr<ppc::core::TaskData> test_data = std::make_shared<ppc::core::TaskData>();
  test_data->inputs_count.emplace_back(rows);
  test_data->inputs_count.emplace_back(cols);

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<unsigned int> expected_image = {0, 0, 0, 0, 0, 6, 7, 0, 0, 10, 11, 0, 0, 0, 0, 0};
  std::vector<unsigned int> expected_image_new(rows * cols);

  sharamygina_i_horizontal_line_filtration_seq::ToFiltSeq(image, rows, cols, expected_image_new);

  test_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

  received_image.resize(rows * cols);
  test_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
  test_data->outputs_count.emplace_back(received_image.size());

  sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq test_task(test_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  ASSERT_EQ(received_image.size(), expected_image.size());
  for (unsigned int i = 0; i < received_image.size(); i++) {
    ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
  }
}

TEST(sharamygina_i_horizontal_line_filtration, BigImageTest) {
  int rows = 200;
  int cols = 160;

  std::shared_ptr<ppc::core::TaskData> test_data = std::make_shared<ppc::core::TaskData>();
  test_data->inputs_count.emplace_back(rows);
  test_data->inputs_count.emplace_back(cols);

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  image = sharamygina_i_horizontal_line_filtration_seq::GetImage(rows, cols);
  sharamygina_i_horizontal_line_filtration_seq::ToFiltSeq(image, rows, cols, expected_image);

  test_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

  received_image.resize(rows * cols);
  test_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
  test_data->outputs_count.emplace_back(received_image.size());

  sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq test_task(test_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  for (unsigned int i = 0; i < received_image.size(); i++) {
    ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
  }
}

TEST(sharamygina_i_horizontal_line_filtration, SmallImageTest) {
  int rows = 5;
  int cols = 4;

  std::shared_ptr<ppc::core::TaskData> test_data = std::make_shared<ppc::core::TaskData>();
  test_data->inputs_count.emplace_back(rows);
  test_data->inputs_count.emplace_back(cols);

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  image = sharamygina_i_horizontal_line_filtration_seq::GetImage(rows, cols);
  test_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

  sharamygina_i_horizontal_line_filtration_seq::ToFiltSeq(image, rows, cols, expected_image);

  received_image.resize(rows * cols);
  test_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
  test_data->outputs_count.emplace_back(received_image.size());

  sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq test_task(test_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  for (unsigned int i = 0; i < received_image.size(); i++) {
    ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
  }
}

TEST(sharamygina_i_horizontal_line_filtration, SquareImageTest) {
  int rows = 5;
  int cols = 5;

  std::shared_ptr<ppc::core::TaskData> test_data = std::make_shared<ppc::core::TaskData>();
  test_data->inputs_count.emplace_back(rows);
  test_data->inputs_count.emplace_back(cols);

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  image = sharamygina_i_horizontal_line_filtration_seq::GetImage(rows, cols);
  sharamygina_i_horizontal_line_filtration_seq::ToFiltSeq(image, rows, cols, expected_image);

  test_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

  received_image.resize(rows * cols);
  test_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
  test_data->outputs_count.emplace_back(received_image.size());

  sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq test_task(test_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  for (unsigned int i = 0; i < received_image.size(); i++) {
    ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
  }
}

TEST(sharamygina_i_horizontal_line_filtration, HorizontalImageTest) {
  int rows = 5;
  int cols = 10;

  std::shared_ptr<ppc::core::TaskData> test_data = std::make_shared<ppc::core::TaskData>();
  test_data->inputs_count.emplace_back(rows);
  test_data->inputs_count.emplace_back(cols);

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  image = sharamygina_i_horizontal_line_filtration_seq::GetImage(rows, cols);
  sharamygina_i_horizontal_line_filtration_seq::ToFiltSeq(image, rows, cols, expected_image);

  test_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

  received_image.resize(rows * cols);
  test_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
  test_data->outputs_count.emplace_back(received_image.size());

  sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq test_task(test_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  for (unsigned int i = 0; i < received_image.size(); i++) {
    ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
  }
}

TEST(sharamygina_i_horizontal_line_filtration, VerticalImageTest) {
  int rows = 10;
  int cols = 5;

  std::shared_ptr<ppc::core::TaskData> test_data = std::make_shared<ppc::core::TaskData>();
  test_data->inputs_count.emplace_back(rows);
  test_data->inputs_count.emplace_back(cols);

  std::vector<unsigned int> received_image;
  std::vector<unsigned int> image(rows * cols);
  std::vector<unsigned int> expected_image(rows * cols);

  image = sharamygina_i_horizontal_line_filtration_seq::GetImage(rows, cols);
  sharamygina_i_horizontal_line_filtration_seq::ToFiltSeq(image, rows, cols, expected_image);

  test_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));

  received_image.resize(rows * cols);
  test_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_image.data()));
  test_data->outputs_count.emplace_back(received_image.size());

  sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq test_task(test_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  for (unsigned int i = 0; i < received_image.size(); i++) {
    ASSERT_EQ(received_image[i], expected_image[i]) << "Difference at i=" << i;
  }
}