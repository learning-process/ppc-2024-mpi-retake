#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/konkov_i_linear_hist_stretch_mpi/include/ops_mpi.hpp"

TEST(konkov_i_linear_hist_stretch_mpi, test_contrast_stretch) {
  std::vector<uint8_t> in = {50, 100, 150, 200, 250};
  std::vector<uint8_t> expected_out = {0, 63, 127, 191, 255};
  std::vector<uint8_t> out(in.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(out, expected_out);
}

TEST(konkov_i_linear_hist_stretch_mpi, test_all_same_values) {
  std::vector<uint8_t> in(5, 128);
  std::vector<uint8_t> expected_out = in;
  std::vector<uint8_t> out(in.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(out, expected_out);
}

TEST(konkov_i_linear_hist_stretch_mpi, test_full_range) {
  std::vector<uint8_t> in = {0, 255};
  std::vector<uint8_t> expected_out = in;
  std::vector<uint8_t> out(in.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(out, expected_out);
}

TEST(konkov_i_linear_hist_stretch_mpi, test_empty_image) {
  std::vector<uint8_t> in;
  std::vector<uint8_t> out;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  konkov_i_linear_hist_stretch_mpi::LinearHistStretchMPI task(task_data);
  ASSERT_FALSE(task.Validation());
}