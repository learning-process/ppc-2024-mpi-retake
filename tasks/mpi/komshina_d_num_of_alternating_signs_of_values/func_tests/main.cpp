#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/komshina_d_num_of_alternating_signs_of_values/include/ops_mpi.hpp"

TEST(komshina_d_num_of_alternations_signs_mpi, NormalCase) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, -1, 1, -1, 1};
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 4);
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, NCase) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, -1, -1, -1, 1};
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 2);
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, InvalidInputs) {
  boost::mpi::communicator world;

  std::vector<int> in = {1};
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.Validation(), false);
}

TEST(komshina_d_num_of_alternations_signs_mpi, MixedSigns) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 0);
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, MaxDataSize) {
  boost::mpi::communicator world;

  std::vector<int> in(10000, 1);
  std::vector<int32_t> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_EQ(out[0], 0);
  }
}

TEST(komshina_d_num_of_alternations_signs_mpi, InvalidOutputSize) {
  boost::mpi::communicator world;

  std::vector<int> in = {1, -1, 1, -1, 1};
  std::vector<int32_t> out(0, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  komshina_d_num_of_alternations_signs_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.Validation(), false);
}