#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"
#include "mpi/makadrai_a_sobel/include/ops_mpi.hpp"

namespace {
std::vector<int> RandomGenerateImg(int height_img, int width_img) {
  std::vector<int> img(height_img * width_img);

  std::random_device rd;
  std::mt19937 gen(static_cast<int>(rd()));
  std::uniform_int_distribution<> ras(0, 255);

  std::ranges::generate(img.begin(), img.end(), [&]() { return ras(gen); });
  return img;
}
}  // namespace

TEST(makadrai_a_sobel_mpi, test_2_2) {
  boost::mpi::communicator world;
  int height_img = 2;
  int width_img = 2;
  std::vector<int> img;
  std::vector<int> res;
  std::vector<int> ans;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img = {100, 50, 150, 200};
    res.resize(4);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

    task_data->inputs_count.emplace_back(width_img);
    task_data->inputs_count.emplace_back(height_img);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(width_img);
    task_data->outputs_count.emplace_back(height_img);

    ans = {228, 255, 201, 175};
  }

  makadrai_a_sobel_mpi::Sobel sobel(task_data);
  ASSERT_TRUE(sobel.Validation());
  sobel.PreProcessing();
  sobel.Run();
  sobel.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(ans, res);
  }
}

TEST(makadrai_a_sobel_mpi, test_3_3) {
  boost::mpi::communicator world;
  int height_img = 3;
  int width_img = 3;
  std::vector<int> img;
  std::vector<int> res;
  std::vector<int> ans;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img = {32, 21, 61, 201, 231, 61, 132, 61, 111};
    res.resize(9);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

    task_data->inputs_count.emplace_back(width_img);
    task_data->inputs_count.emplace_back(height_img);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(width_img);
    task_data->outputs_count.emplace_back(height_img);

    ans = {235, 248, 152, 203, 121, 191, 247, 255, 170};
  }

  makadrai_a_sobel_mpi::Sobel sobel(task_data);
  ASSERT_TRUE(sobel.Validation());
  sobel.PreProcessing();
  sobel.Run();
  sobel.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(ans, res);
  }
}

TEST(makadrai_a_sobel_mpi, test_1_1) {
  boost::mpi::communicator world;
  int height_img = 1;
  int width_img = 1;
  std::vector<int> img;
  std::vector<int> res;
  std::vector<int> ans;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img = {32};
    res.resize(1);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

    task_data->inputs_count.emplace_back(width_img);
    task_data->inputs_count.emplace_back(height_img);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(width_img);
    task_data->outputs_count.emplace_back(height_img);

    ans = {0};
  }

  makadrai_a_sobel_mpi::Sobel sobel(task_data);
  ASSERT_TRUE(sobel.Validation());
  sobel.PreProcessing();
  sobel.Run();
  sobel.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(ans, res);
  }
}

TEST(makadrai_a_sobel_mpi, test_error_1) {
  boost::mpi::communicator world;
  int height_img = 3;
  int width_img = 3;
  std::vector<int> img;
  std::vector<int> res;
  std::vector<int> ans;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img = {32, 21, 61, 201, 231, 61, 132, 61, 111};
    res.resize(9);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

    task_data->inputs_count.emplace_back(width_img);
    task_data->inputs_count.emplace_back(height_img);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(width_img + 1);
    task_data->outputs_count.emplace_back(height_img);

    ans = {235, 248, 152, 203, 121, 191, 247, 255, 170};
  }

  makadrai_a_sobel_mpi::Sobel sobel(task_data);
  ASSERT_FALSE(sobel.Validation());
}

TEST(makadrai_a_sobel_mpi, test_error_2) {
  boost::mpi::communicator world;
  int height_img = 3;
  int width_img = 3;
  std::vector<int> img;
  std::vector<int> res;
  std::vector<int> ans;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img = {32, 21, 61, 201, 231, 61, 132, 61, 111};
    res.resize(9);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

    task_data->inputs_count.emplace_back(width_img + 1);
    task_data->inputs_count.emplace_back(height_img);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(width_img);
    task_data->outputs_count.emplace_back(height_img);

    ans = {235, 248, 152, 203, 121, 191, 247, 255, 170};
  }

  makadrai_a_sobel_mpi::Sobel sobel(task_data);
  ASSERT_FALSE(sobel.Validation());
}

TEST(makadrai_a_sobel_mpi, test_error_3) {
  boost::mpi::communicator world;
  int height_img = 3;
  int width_img = 3;
  std::vector<int> img;
  std::vector<int> res;
  std::vector<int> ans;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img = {32, 21, 61, 201, 231, 61, 132, 61, 111};
    res.resize(8);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

    task_data->inputs_count.emplace_back(width_img);
    task_data->inputs_count.emplace_back(height_img);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(width_img + 1);
    task_data->outputs_count.emplace_back(height_img);

    ans = {235, 248, 152, 203, 121, 191, 247, 255, 170};
  }

  makadrai_a_sobel_mpi::Sobel sobel(task_data);
  ASSERT_FALSE(sobel.Validation());
}

TEST(makadrai_a_sobel_mpi, test_random_100_100) {
  boost::mpi::communicator world;
  int height_img = 100;
  int width_img = 100;
  std::vector<int> img;
  std::vector<int> res;
  std::vector<int> ans;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img = RandomGenerateImg(height_img, width_img);
    res.resize(height_img * width_img);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

    task_data->inputs_count.emplace_back(width_img);
    task_data->inputs_count.emplace_back(height_img);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(width_img);
    task_data->outputs_count.emplace_back(height_img);
  }

  makadrai_a_sobel_mpi::Sobel sobel(task_data);
  ASSERT_TRUE(sobel.Validation());
  sobel.PreProcessing();
  sobel.Run();
  sobel.PostProcessing();

  if (world.rank() == 0) {
    ans.resize(width_img * height_img);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

    task_data_seq->inputs_count.emplace_back(width_img);
    task_data_seq->inputs_count.emplace_back(height_img);

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    task_data_seq->outputs_count.emplace_back(width_img);
    task_data_seq->outputs_count.emplace_back(height_img);

    makadrai_a_sobel_mpi::SobelSeq sobel_seq(task_data_seq);
    ASSERT_TRUE(sobel_seq.Validation());
    sobel_seq.PreProcessing();
    sobel_seq.Run();
    sobel_seq.PostProcessing();

    EXPECT_EQ(ans, res);
  }
}

TEST(makadrai_a_sobel_mpi, test_random_100_50) {
  boost::mpi::communicator world;
  int height_img = 100;
  int width_img = 50;
  std::vector<int> img;
  std::vector<int> res;
  std::vector<int> ans;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img = RandomGenerateImg(height_img, width_img);
    res.resize(height_img * width_img);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

    task_data->inputs_count.emplace_back(width_img);
    task_data->inputs_count.emplace_back(height_img);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(width_img);
    task_data->outputs_count.emplace_back(height_img);
  }

  makadrai_a_sobel_mpi::Sobel sobel(task_data);
  ASSERT_TRUE(sobel.Validation());
  sobel.PreProcessing();
  sobel.Run();
  sobel.PostProcessing();

  if (world.rank() == 0) {
    ans.resize(width_img * height_img);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

    task_data_seq->inputs_count.emplace_back(width_img);
    task_data_seq->inputs_count.emplace_back(height_img);

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
    task_data_seq->outputs_count.emplace_back(width_img);
    task_data_seq->outputs_count.emplace_back(height_img);

    makadrai_a_sobel_mpi::SobelSeq sobel_seq(task_data_seq);
    ASSERT_TRUE(sobel_seq.Validation());
    sobel_seq.PreProcessing();
    sobel_seq.Run();
    sobel_seq.PostProcessing();

    EXPECT_EQ(ans, res);
  }
}