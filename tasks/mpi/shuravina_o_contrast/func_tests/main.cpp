#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

TEST(shuravina_o_contrast, test_min_max_values) {
  constexpr size_t kCount = 256;

  std::vector<uint8_t> in(kCount * kCount, 0);
  std::vector<uint8_t> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
      in[(i * kCount) + j] = static_cast<uint8_t>(i + j);
    }
  }

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  uint8_t max_val = *std::ranges::max_element(out);
  EXPECT_EQ(max_val, 255);
}

TEST(shuravina_o_contrast, test_random_values) {
  constexpr size_t kCount = 256;

  std::vector<uint8_t> in(kCount * kCount, 0);
  std::vector<uint8_t> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
      in[(i * kCount) + j] = static_cast<uint8_t>(rand() % 256);
    }
  }

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  uint8_t min_val = *std::ranges::min_element(out);
  uint8_t max_val = *std::ranges::max_element(out);
  EXPECT_EQ(min_val, 0);
  EXPECT_EQ(max_val, 255);
}
TEST(shuravina_o_contrast, test_all_values_same) {
  constexpr size_t kCount = 256;

  std::vector<uint8_t> in(kCount * kCount, 100);
  std::vector<uint8_t> out(kCount * kCount, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  uint8_t max_val = *std::ranges::max_element(out);
  uint8_t min_val = *std::ranges::min_element(out);
  EXPECT_EQ(max_val, 255);
  EXPECT_EQ(min_val, 255);
}

TEST(shuravina_o_contrast, test_all_values_max) {
  constexpr size_t kCount = 256;

  std::vector<uint8_t> in(kCount * kCount, 255);
  std::vector<uint8_t> out(kCount * kCount, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  uint8_t max_val = *std::ranges::max_element(out);
  uint8_t min_val = *std::ranges::min_element(out);
  EXPECT_EQ(max_val, 255);
  EXPECT_EQ(min_val, 255);
}

TEST(shuravina_o_contrast, test_alternating_min_max_values) {
  constexpr size_t kCount = 256;

  std::vector<uint8_t> in(kCount * kCount, 0);
  std::vector<uint8_t> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
      in[(i * kCount) + j] = (i + j) % 2 == 0 ? 0 : 255;
    }
  }

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  uint8_t max_val = *std::ranges::max_element(out);
  uint8_t min_val = *std::ranges::min_element(out);
  EXPECT_EQ(max_val, 255);
  EXPECT_EQ(min_val, 0);
}

TEST(shuravina_o_contrast, test_single_unique_value) {
  constexpr size_t kCount = 256;

  std::vector<uint8_t> in(kCount * kCount, 128);
  std::vector<uint8_t> out(kCount * kCount, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  uint8_t max_val = *std::ranges::max_element(out);
  uint8_t min_val = *std::ranges::min_element(out);
  EXPECT_EQ(max_val, 255);
  EXPECT_EQ(min_val, 255);
}