#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/karaseva_e_binaryimage/include/ops_seq.hpp"

TEST(karaseva_e_binaryimage_seq, test_img_is_object) {
  const int rows = 3;
  const int cols = 3;                                 // 3x3 binary image
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // Entire image is an object
  std::vector<int> out(rows * cols, -1);
  std::vector<int> expected_out(rows * cols, 2);  // Single labeled component

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count = {rows, cols};
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count = {rows, cols};

  karaseva_e_binaryimage_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(expected_out, out);
}

TEST(karaseva_e_binaryimage_seq, test_img_is_background) {
  const int rows = 3;
  const int cols = 3;                                 // 3x3 binary image
  std::vector<int> in = {1, 1, 1, 1, 1, 1, 1, 1, 1};  // Entire image is background
  std::vector<int> out(rows * cols, -1);
  std::vector<int> expected_out(rows * cols, 1);  // Background remains 1

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count = {rows, cols};
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count = {rows, cols};

  karaseva_e_binaryimage_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(expected_out, out);
}

TEST(karaseva_e_binaryimage_seq, test_two_separated_objects) {
  const int rows = 3;
  const int cols = 3;
  std::vector<int> in = {0, 1, 0, 1, 1, 1, 0, 1, 0};
  std::vector<int> out(rows * cols, -1);
  std::vector<int> expected_out = {2, 1, 3, 1, 1, 1, 4, 1, 5};  // Two objects with labels 2, 3, 4, 5

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count = {rows, cols};
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count = {rows, cols};

  karaseva_e_binaryimage_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(expected_out, out);
}

TEST(karaseva_e_binaryimage_seq, test_complex_object_shape) {
  const int rows = 4;
  const int cols = 4;
  std::vector<int> in = {1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0};
  std::vector<int> out(rows * cols, -1);
  std::vector<int> expected_out = {1, 2, 2, 1, 1, 2, 1, 1,
                                   1, 2, 2, 2, 1, 1, 1, 2};  // Object labeled with different labels

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count = {rows, cols};
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count = {rows, cols};

  karaseva_e_binaryimage_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(expected_out, out);
}