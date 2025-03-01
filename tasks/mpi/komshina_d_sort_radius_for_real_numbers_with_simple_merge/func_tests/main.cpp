#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, VerifySortingWithPreGeneratedData) {
  std::vector<double> testData = {11.5, 3.3, 5.7, 9.0, 2.2, 4.5, 8.8, 7.1, 6.1};
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dataSize = testData.size();
  std::vector<double> parallelResult(dataSize, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(testData.data()));
    task_data_mpi->inputs_count.emplace_back(dataSize);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallelResult.data()));
    task_data_mpi->outputs_count.emplace_back(dataSize);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (rank == 0) {
    for (int i = 1; i < dataSize; ++i) {
      ASSERT_LE(parallelResult[i - 1], parallelResult[i]);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, HandlesNegativeNumbers) {
  std::vector<double> testData = {-10.5, -2.3, -4.7, -8.0, -1.2, -3.5, -7.8, -6.1, -5.1};
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dataSize = testData.size();
  std::vector<double> parallelResult(dataSize, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(testData.data()));
    task_data_mpi->inputs_count.emplace_back(dataSize);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallelResult.data()));
    task_data_mpi->outputs_count.emplace_back(dataSize);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (rank == 0) {
    for (int i = 1; i < dataSize; ++i) {
      ASSERT_LE(parallelResult[i - 1], parallelResult[i]);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, HandlesMixedNumbers) {
  std::vector<double> testData = {10.5, -2.3, 4.7, -8.0, 1.2, -3.5, 7.8, -6.1, 5.1};
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dataSize = testData.size();
  std::vector<double> parallelResult(dataSize, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(testData.data()));
    task_data_mpi->inputs_count.emplace_back(dataSize);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallelResult.data()));
    task_data_mpi->outputs_count.emplace_back(dataSize);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (rank == 0) {
    for (int i = 1; i < dataSize; ++i) {
      ASSERT_LE(parallelResult[i - 1], parallelResult[i]);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, HandlesEmptyInput) {
  std::vector<double> testData = {};
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dataSize = testData.size();
  std::vector<double> parallelResult(dataSize, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(testData.data()));
    task_data_mpi->inputs_count.emplace_back(dataSize);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallelResult.data()));
    task_data_mpi->outputs_count.emplace_back(dataSize);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (rank == 0) {
    ASSERT_EQ(parallelResult.size(), 0);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, HandlesIdenticalElements) {
  std::vector<double> testData = {5.5, 5.5, 5.5, 5.5, 5.5};
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dataSize = testData.size();
  std::vector<double> parallelResult(dataSize, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(testData.data()));
    task_data_mpi->inputs_count.emplace_back(dataSize);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallelResult.data()));
    task_data_mpi->outputs_count.emplace_back(dataSize);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (rank == 0) {
    for (int i = 1; i < dataSize; ++i) {
      ASSERT_NEAR(parallelResult[i], parallelResult[i - 1], 1e-10);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, HandlesSingleElement) {
  std::vector<double> testData = {5.5};
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dataSize = testData.size();
  std::vector<double> parallelResult(dataSize, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(testData.data()));
    task_data_mpi->inputs_count.emplace_back(dataSize);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallelResult.data()));
    task_data_mpi->outputs_count.emplace_back(dataSize);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (rank == 0) {
    ASSERT_EQ(parallelResult[0], 5.5);
  }
}