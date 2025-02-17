#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <vector>

#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

void FillInputImage(std::vector<uint8_t>& in, size_t kSize);

void FillInputImage(std::vector<uint8_t>& in, size_t kSize) {
  for (size_t i = 0; i < kSize; ++i) {
    for (size_t j = 0; j < kSize; ++j) {
      in[(i * kSize) + j] = static_cast<uint8_t>(i + j);
    }
  }
}

TEST(shuravina_o_contrast, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  constexpr size_t kSize = 256;
  std::vector<uint8_t> in(kSize * kSize, 0);
  std::vector<uint8_t> out(kSize * kSize, 0);

  FillInputImage(in, kSize);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_TRUE(test_task_mpi.Validation());
  ASSERT_TRUE(test_task_mpi.PreProcessing());
  ASSERT_TRUE(test_task_mpi.Run());
  ASSERT_TRUE(test_task_mpi.PostProcessing());

  if (world.rank() == 0) {
    uint8_t max_val = *std::ranges::max_element(out);
    EXPECT_EQ(max_val, 255);
  }
}