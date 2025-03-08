// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/makhov_m_ring_topology/include/ops_mpi.hpp"

TEST(makhov_m_ring_topology, VectorZeroSize) {
  boost::mpi::communicator world;
  size_t size = 0;
  std::vector<int32_t> input_vector(size);
  std::vector<int32_t> output_vector(size);
  std::vector<int32_t> sequence(world.size() + 1);
  std::vector<int32_t> reference_sequence(world.size() + 1);

  for (int32_t i = 0; i < static_cast<int32_t>(world.size()); i++) {
    reference_sequence[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequence.data()));
    task_data_par->outputs_count.emplace_back(size);
    task_data_par->outputs_count.emplace_back(world.size() + 1);
  }

  // Create Task
  makhov_m_ring_topology::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel.ValidationImpl());
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(input_vector, output_vector);
    ASSERT_EQ(sequence, reference_sequence);
  }
}

TEST(makhov_m_ring_topology, RandVectorSize1) {
  size_t size = 1;
  std::vector<int32_t> input_vector(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int32_t> res(size);
  for (size_t i = 0; i < size; i++) {
    input_vector[i] = static_cast<int32_t>((0 + (gen() % 10)));
  }

  boost::mpi::communicator world;

  std::vector<int32_t> output_vector(size);
  std::vector<int32_t> sequence(world.size() + 1);
  std::vector<int32_t> reference_sequence(world.size() + 1);

  for (int32_t i = 0; i < static_cast<int32_t>(world.size()); i++) {
    reference_sequence[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequence.data()));
    task_data_par->outputs_count.emplace_back(size);
    task_data_par->outputs_count.emplace_back(world.size() + 1);
  }

  // Create Task
  makhov_m_ring_topology::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel.ValidationImpl());
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(input_vector, output_vector);
    ASSERT_EQ(sequence, reference_sequence);
  }
}

TEST(makhov_m_ring_topology, RandVectorSize10) {
  size_t size = 10;
  std::vector<int32_t> input_vector(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int32_t> res(size);
  for (size_t i = 0; i < size; i++) {
    input_vector[i] = static_cast<int32_t>((0 + (gen() % 10)));
  }

  boost::mpi::communicator world;

  std::vector<int32_t> output_vector(size);
  std::vector<int32_t> sequence(world.size() + 1);
  std::vector<int32_t> reference_sequence(world.size() + 1);

  for (int32_t i = 0; i < static_cast<int32_t>(world.size()); i++) {
    reference_sequence[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequence.data()));
    task_data_par->outputs_count.emplace_back(size);
    task_data_par->outputs_count.emplace_back(world.size() + 1);
  }

  // Create Task
  makhov_m_ring_topology::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel.ValidationImpl());
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(input_vector, output_vector);
    ASSERT_EQ(sequence, reference_sequence);
  }
}

TEST(makhov_m_ring_topology, RandVectorSize1000) {
  size_t size = 1000;
  std::vector<int32_t> input_vector(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int32_t> res(size);
  for (size_t i = 0; i < size; i++) {
    input_vector[i] = static_cast<int32_t>((0 + (gen() % 10)));
  }

  boost::mpi::communicator world;

  std::vector<int32_t> output_vector(size);
  std::vector<int32_t> sequence(world.size() + 1);
  std::vector<int32_t> reference_sequence(world.size() + 1);

  for (int32_t i = 0; i < static_cast<int32_t>(world.size()); i++) {
    reference_sequence[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequence.data()));
    task_data_par->outputs_count.emplace_back(size);
    task_data_par->outputs_count.emplace_back(world.size() + 1);
  }

  // Create Task
  makhov_m_ring_topology::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel.ValidationImpl());
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(input_vector, output_vector);
    ASSERT_EQ(sequence, reference_sequence);
  }
}
