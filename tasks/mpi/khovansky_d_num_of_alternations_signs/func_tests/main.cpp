#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>
#include <stdint.h>

#include <boost/mpi/communicator.hpp>
#include "core/task/include/task.hpp"
#include "mpi/khovansky_d_num_of_alternations_signs/include/ops_mpi.hpp"

TEST(khovansky_d_num_of_alternations_signs_mpi, test_10) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in = {1, 2, -3, -4, -5, 6, -7, 8, 9, 10};
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  // Create Task
  khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi num_of_alternations_signs_mpi(task_data_mpi);
  ASSERT_EQ(num_of_alternations_signs_mpi.ValidationImpl(), true);
  num_of_alternations_signs_mpi.PreProcessingImpl();
  num_of_alternations_signs_mpi.RunImpl();
  num_of_alternations_signs_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(4, out[0]);
  }
}

TEST(khovansky_d_num_of_alternations_signs_mpi, invalid_input) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in = {1};
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  // Create Task
  khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi num_of_alternations_signs_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(num_of_alternations_signs_mpi.ValidationImpl(), false);
  }
}

TEST(khovansky_d_num_of_alternations_signs_mpi, test_with_zero) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in = {1, 0, -1, 0, 0, -1, -1, 1};
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  // Create Task
  khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi num_of_alternations_signs_mpi(task_data_mpi);
  ASSERT_EQ(num_of_alternations_signs_mpi.ValidationImpl(), true);
  num_of_alternations_signs_mpi.PreProcessingImpl();
  num_of_alternations_signs_mpi.RunImpl();
  num_of_alternations_signs_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(4, out[0]);
  }
}

TEST(khovansky_d_num_of_alternations_signs_mpi, test_with_only_zero) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in = {0, 0, 0, 0, 0, 0};
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  // Create Task
  khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi num_of_alternations_signs_mpi(task_data_mpi);
  ASSERT_EQ(num_of_alternations_signs_mpi.ValidationImpl(), true);
  num_of_alternations_signs_mpi.PreProcessingImpl();
  num_of_alternations_signs_mpi.RunImpl();
  num_of_alternations_signs_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    ASSERT_EQ(0, out[0]);
  }
}

TEST(khovansky_d_num_of_alternations_signs_mpi, test_with_only_positive) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in = {1, 0, 1, 0, 0, 1, 1, 1};
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  // Create Task
  khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi num_of_alternations_signs_mpi(task_data_mpi);
  ASSERT_EQ(num_of_alternations_signs_mpi.ValidationImpl(), true);
  num_of_alternations_signs_mpi.PreProcessingImpl();
  num_of_alternations_signs_mpi.RunImpl();
  num_of_alternations_signs_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    ASSERT_EQ(0, out[0]);
  }
}

TEST(khovansky_d_num_of_alternations_signs_mpi, test_with_only_negative) {
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in = {-1, -1, -1};
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  // Create Task
  khovansky_d_num_of_alternations_signs_mpi::NumOfAlternationsSignsMpi num_of_alternations_signs_mpi(task_data_mpi);
  ASSERT_EQ(num_of_alternations_signs_mpi.ValidationImpl(), true);
  num_of_alternations_signs_mpi.PreProcessingImpl();
  num_of_alternations_signs_mpi.RunImpl();
  num_of_alternations_signs_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    ASSERT_EQ(0, out[0]);
  }
}