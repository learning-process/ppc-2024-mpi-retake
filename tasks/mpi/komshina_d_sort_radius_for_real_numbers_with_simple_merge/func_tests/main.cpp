#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, HandlesEmptyInput) {
  std::vector<double> test_data = {};
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int data_size = static_cast<int>(test_data.size());
  std::vector<double> parallel_result(data_size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world_rank == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_data.data()));
    task_data_mpi->inputs_count.emplace_back(data_size);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_result.data()));
    task_data_mpi->outputs_count.emplace_back(data_size);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world_rank == 0) {
    ASSERT_EQ(static_cast<int>(parallel_result.size()), 0);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, HandlesIdenticalElements) {
  std::vector<double> test_data = {5.5, 5.5, 5.5, 5.5, 5.5};
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int data_size = static_cast<int>(test_data.size());
  std::vector<double> parallel_result(data_size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world_rank == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_data.data()));
    task_data_mpi->inputs_count.emplace_back(data_size);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_result.data()));
    task_data_mpi->outputs_count.emplace_back(data_size);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world_rank == 0) {
    for (int i = 1; i < data_size; ++i) {
      ASSERT_NEAR(parallel_result[i], parallel_result[i - 1], 1e-10);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, HandlesSingleElement) {
  std::vector<double> test_data = {5.5};
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int data_size = static_cast<int>(test_data.size());
  std::vector<double> parallel_result(data_size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world_rank == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_data.data()));
    task_data_mpi->inputs_count.emplace_back(data_size);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_result.data()));
    task_data_mpi->outputs_count.emplace_back(data_size);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world_rank == 0) {
    ASSERT_EQ(parallel_result[0], 5.5);
  }
}