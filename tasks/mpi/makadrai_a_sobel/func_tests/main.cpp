#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/makadrai_a_sobel/include/ops_mpi.hpp"

TEST(makadrai_a_sobel_mpi, test_2_2) {
  boost::mpi::communicator world;
  size_t height_img = 2;
  size_t width_img = 2;
  std::vector<size_t> img;
  std::vector<size_t> res;
  std::vector<size_t> ans;

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
  size_t height_img = 3;
  size_t width_img = 3;
  std::vector<size_t> img;
  std::vector<size_t> res;
  std::vector<size_t> ans;

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
  size_t height_img = 1;
  size_t width_img = 1;
  std::vector<size_t> img;
  std::vector<size_t> res;
  std::vector<size_t> ans;

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
  size_t height_img = 3;
  size_t width_img = 3;
  std::vector<size_t> img;
  std::vector<size_t> res;
  std::vector<size_t> ans;

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
  size_t height_img = 3;
  size_t width_img = 3;
  std::vector<size_t> img;
  std::vector<size_t> res;
  std::vector<size_t> ans;

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
  size_t height_img = 3;
  size_t width_img = 3;
  std::vector<size_t> img;
  std::vector<size_t> res;
  std::vector<size_t> ans;

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