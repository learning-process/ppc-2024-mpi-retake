#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "seq/solovev_a_binary_image_marking/include/ops_sec.hpp"

namespace {

void TestBodyFunction(const int m, const int n, std::vector<int> data, std::vector<int> exp_image) {
  std::vector<int> labledImage(m * n);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&m)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&n)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(data.data())));
  taskDataSeq->inputs_count.emplace_back(data.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(labledImage.data()));
  taskDataSeq->outputs_count.emplace_back(labledImage.size());

  solovev_a_binary_image_marking::TestTaskSequential BinaryImage(taskDataSeq);

  ASSERT_TRUE(BinaryImage.ValidationImpl());
  ASSERT_TRUE(BinaryImage.PreProcessingImpl());
  ASSERT_TRUE(BinaryImage.RunImpl());
  ASSERT_TRUE(BinaryImage.PostProcessingImpl());

  std::cout << std::endl;
  for (size_t i = 0; i < exp_image.size(); ++i) {
    ASSERT_EQ(labledImage[i], exp_image[i]);
  }
}
}  // namespace

TEST(solovev_a_binary_image_marking, single_pixel) {
  int m = 1;
  int n = 1;

  std::vector<int> data = {1};
  std::vector<int> exp_image = {1};

  TestBodyFunction(m, n, data, exp_image);
}

TEST(solovev_a_binary_image_marking, two_components) {
  int m = 3;
  int n = 1;

  std::vector<int> data = {1, 0, 1};
  std::vector<int> exp_image = {1, 0, 2};

  TestBodyFunction(m, n, data, exp_image);
}

TEST(solovev_a_binary_image_marking, v_single_line) {
  int m = 1;
  int n = 3;

  std::vector<int> data = {1, 1, 1};
  std::vector<int> exp_image = {1, 1, 1};

  TestBodyFunction(m, n, data, exp_image);
}

TEST(solovev_a_binary_image_marking, h_single_line) {
  int m = 3;
  int n = 1;

  std::vector<int> data = {1, 1, 1};
  std::vector<int> exp_image = {1, 1, 1};

  TestBodyFunction(m, n, data, exp_image);
}

TEST(solovev_a_binary_image_marking, cross) {
  int m = 3;
  int n = 3;

  std::vector<int> data = {0, 1, 0, 1, 1, 1, 0, 1, 0};

  std::vector<int> exp_image = {0, 1, 0, 1, 1, 1, 0, 1, 0};

  TestBodyFunction(m, n, data, exp_image);
}

TEST(solovev_a_binary_image_marking, four_components) {
  int m = 4;
  int n = 5;

  std::vector<int> data = {1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1};

  std::vector<int> exp_image = {1, 1, 0, 0, 2, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 4, 4};

  TestBodyFunction(m, n, data, exp_image);
}

TEST(solovev_a_binary_image_marking, x_cross) {
  int m = 3;
  int n = 3;

  std::vector<int> data = {1, 0, 1, 0, 1, 0, 1, 0, 1};

  std::vector<int> exp_image = {1, 0, 2, 0, 3, 0, 4, 0, 5};

  TestBodyFunction(m, n, data, exp_image);
}

TEST(solovev_a_binary_image_marking, square) {
  int m = 4;
  int n = 4;

  std::vector<int> data = {1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1};

  std::vector<int> exp_image = {1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1};

  TestBodyFunction(m, n, data, exp_image);
}

TEST(solovev_a_binary_image_marking, lots_of_components) {
  int m = 5;
  int n = 6;

  std::vector<int> data = {1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1};

  std::vector<int> exp_image = {1, 0, 2, 2, 0, 2, 1, 0, 0, 2, 0, 2, 1, 1, 0,
                                2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 4, 4};

  TestBodyFunction(m, n, data, exp_image);
}

TEST(solovev_a_binary_image_marking, validation_false_1) {
  int m = -1;
  int n = 1;

  std::vector<int> data = {1};
  std::vector<int> labledImage = {1};

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&m)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&n)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(data.data())));
  taskDataSeq->inputs_count.emplace_back(data.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(labledImage.data()));
  taskDataSeq->outputs_count.emplace_back(labledImage.size());

  solovev_a_binary_image_marking::TestTaskSequential BinaryImage(taskDataSeq);

  ASSERT_FALSE(BinaryImage.ValidationImpl());
}

TEST(solovev_a_binary_image_marking, validation_false_2) {
  int m = 1;
  int n = -1;

  std::vector<int> data = {1};
  std::vector<int> labledImage = {1};

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&m)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&n)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(data.data())));
  taskDataSeq->inputs_count.emplace_back(data.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(labledImage.data()));
  taskDataSeq->outputs_count.emplace_back(labledImage.size());

  solovev_a_binary_image_marking::TestTaskSequential BinaryImage(taskDataSeq);

  ASSERT_FALSE(BinaryImage.ValidationImpl());
}

TEST(solovev_a_binary_image_marking, validation_false_3) {
  int m = 1;
  int n = 1;

  std::vector<int> data = {};
  std::vector<int> labledImage = {};

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&m)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&n)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(data.data())));
  taskDataSeq->inputs_count.emplace_back(data.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(labledImage.data()));
  taskDataSeq->outputs_count.emplace_back(labledImage.size());

  solovev_a_binary_image_marking::TestTaskSequential BinaryImage(taskDataSeq);

  ASSERT_FALSE(BinaryImage.ValidationImpl());
}
