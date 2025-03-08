#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/karaseva_e_num_of_alternations_signs/include/ops_mpi.hpp"

TEST(karaseva_e_num_of_alternations_signs_mpi, test_alternations) {
  boost::mpi::communicator world;

  size_t k_count = 50;
  std::vector<int> in(k_count, 0);
  std::vector<int> out(1, 0);

  for (size_t i = 0; i < k_count; i++) {
    in[i] = (i % 2 == 0) ? 1 : -1;
  }

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  karaseva_e_num_of_alternations_signs_mpi::AlternatingSignsMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(out[0], static_cast<int>(k_count - 1));
  }
}

TEST(karaseva_e_num_of_alternations_signs_mpi, test_10) {
  boost::mpi::communicator world;

  std::vector<int> in = {-1, 2, -3, 4, -5, 6, -7, 8, 9, -10};
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  karaseva_e_num_of_alternations_signs_mpi::AlternatingSignsMPI num_of_alternations_signs_mpi(task_data_mpi);
  ASSERT_EQ(num_of_alternations_signs_mpi.ValidationImpl(), true);
  num_of_alternations_signs_mpi.PreProcessingImpl();
  num_of_alternations_signs_mpi.RunImpl();
  num_of_alternations_signs_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 8);
  }
}

TEST(karaseva_e_num_of_alternations_signs_mpi, test_all_positive) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  karaseva_e_num_of_alternations_signs_mpi::AlternatingSignsMPI num_of_alternations_signs_mpi(task_data_mpi);
  ASSERT_EQ(num_of_alternations_signs_mpi.ValidationImpl(), true);
  num_of_alternations_signs_mpi.PreProcessingImpl();
  num_of_alternations_signs_mpi.RunImpl();
  num_of_alternations_signs_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(0, out[0]);
  }
}

TEST(karaseva_e_num_of_alternations_signs_mpi, test_alternating_signs) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  karaseva_e_num_of_alternations_signs_mpi::AlternatingSignsMPI num_of_alternations_signs_mpi(task_data_mpi);
  ASSERT_EQ(num_of_alternations_signs_mpi.ValidationImpl(), true);
  num_of_alternations_signs_mpi.PreProcessingImpl();
  num_of_alternations_signs_mpi.RunImpl();
  num_of_alternations_signs_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(9, out[0]);
  }
}