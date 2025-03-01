#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi {

void generate_random_data(std::vector<double>& data, int N, double min = -1e9, double max = 1e9) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min, max);

  data.resize(N);
  for (int i = 0; i < N; ++i) {
    data[i] = dist(gen);
  }
}

}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, test_pipeline_run) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N = 500000;
  std::vector<double> inputData(N);
  std::vector<double> sortedData(N, 0.0);

  if (rank == 0) {
    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::generate_random_data(inputData, N);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskData->inputs_count.emplace_back(N);

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(sortedData.data()));
    taskData->outputs_count.emplace_back(N);
  }

  auto mpiTask = std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI>(taskData);

  ASSERT_TRUE(mpiTask->ValidationImpl()) << "Validation failed!";
  mpiTask->PreProcessingImpl();
  mpiTask->RunImpl();
  mpiTask->PostProcessingImpl();

  if (rank == 0) {
    std::vector<double> referenceData = inputData;
    std::sort(referenceData.begin(), referenceData.end());
    ASSERT_EQ(referenceData, sortedData) << "MPI sorting is incorrect!";
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, test_task_run) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N = 500000;
  std::vector<double> inputData;
  std::vector<double> sortedData(N, 0.0);

  if (rank == 0) {
    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::generate_random_data(inputData, N);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskData->inputs_count.emplace_back(N);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(sortedData.data()));
    taskData->outputs_count.emplace_back(N);
  }

  auto mpiTask = std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI>(taskData);

  ASSERT_TRUE(mpiTask->ValidationImpl()) << "Validation failed!";
  mpiTask->PreProcessingImpl();
  mpiTask->RunImpl();
  mpiTask->PostProcessingImpl();

  if (rank == 0) {
    std::vector<double> referenceData = inputData;
    std::sort(referenceData.begin(), referenceData.end());
    ASSERT_EQ(referenceData, sortedData) << "MPI sorting is incorrect!";
  }
}
