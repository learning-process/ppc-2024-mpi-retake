#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

TEST(karaseva_e_reduce_mpi, test_reduce_large_matrix) {
  MPI_Comm world = MPI_COMM_WORLD;
  int rank = 0, size = 0;
  MPI_Comm_rank(world, &rank);
  MPI_Comm_size(world, &size);

  constexpr size_t N = 1000;
  std::vector<int> local_data(N, rank + 1);
  std::vector<int> reduced_data(N, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(local_data.data()));
  task_data_mpi->inputs_count.emplace_back(local_data.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(reduced_data.data()));
  task_data_mpi->outputs_count.emplace_back(reduced_data.size());

  // Reduce
  karaseva_e_reduce_mpi::TestTaskMPI test_task_mpi(task_data_mpi, local_data.size());
  ASSERT_TRUE(test_task_mpi.Validation());
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  // Checking the result
  if (rank == 0) {
    std::vector<int> expected_result(N, 0);
    for (int p = 0; p < size; ++p) {
      for (size_t i = 0; i < N; ++i) {
        expected_result[i] += (p + 1);
      }
    }
    EXPECT_EQ(reduced_data, expected_result);
  }
}

TEST(karaseva_e_reduce_mpi, test_reduce_vs_mpi_reduce) {
  MPI_Comm world = MPI_COMM_WORLD;
  int rank = 0, size = 0;
  MPI_Comm_rank(world, &rank);
  MPI_Comm_size(world, &size);

  constexpr size_t N = 1000;
  std::vector<int> local_data(N, rank + 1);
  std::vector<int> custom_reduce_result(N, 0);
  std::vector<int> mpi_reduce_result(N, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(local_data.data()));
  task_data_mpi->inputs_count.emplace_back(local_data.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(custom_reduce_result.data()));
  task_data_mpi->outputs_count.emplace_back(custom_reduce_result.size());

  // Reduce
  karaseva_e_reduce_mpi::TestTaskMPI test_task_mpi(task_data_mpi, local_data.size());
  ASSERT_TRUE(test_task_mpi.Validation());
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  // MPI_Reduce
  MPI_Reduce(local_data.data(), mpi_reduce_result.data(), N, MPI_INT, MPI_SUM, 0, world);

  // Checking the result
  if (rank == 0) {
    EXPECT_EQ(custom_reduce_result, mpi_reduce_result);
  }
}