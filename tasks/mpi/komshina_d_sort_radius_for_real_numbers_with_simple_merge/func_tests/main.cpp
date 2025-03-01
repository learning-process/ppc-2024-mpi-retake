#include <gtest/gtest.h>
#include <mpi.h>

#include <random>
#include <vector>

#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi {

std::vector<double> generateRandomData(int size, double minValue, double maxValue) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(minValue, maxValue);

  std::vector<double> data(size);
  for (int i = 0; i < size; ++i) {
    data[i] = dis(gen);
  }
  return data;
}

void runSortingTest(const std::vector<double>& testData) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dataSize = testData.size();
  std::vector<double> parallelResult(dataSize, 0.0);

  auto parallelTaskData = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(testData.data())));
    parallelTaskData->inputs_count.emplace_back(dataSize);
    parallelTaskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallelResult.data()));
    parallelTaskData->outputs_count.emplace_back(dataSize);
  }

  TestTaskMPI parallelSortTask(parallelTaskData);
  ASSERT_TRUE(parallelSortTask.ValidationImpl());
  parallelSortTask.PreProcessingImpl();
  parallelSortTask.RunImpl();
  parallelSortTask.PostProcessingImpl();

  if (rank == 0) {
    std::vector<double> referenceData = testData;
    std::sort(referenceData.begin(), referenceData.end());
    ASSERT_EQ(referenceData, parallelResult) << "MPI sorting is incorrect!";
  }
}

}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, VerifySorting) {
  const int dataSize = 100;
  const double minValue = -1000.0;
  const double maxValue = 1000.0;
  std::vector<double> testData =
      komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::generateRandomData(dataSize, minValue, maxValue);
  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::runSortingTest(testData);
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, VerifySortingWithPreGeneratedData) {
  std::vector<double> testData = {10.5, -2.3, 4.7, 8.0, -1.2, 3.5, 7.8, -6.1, 5.1};
  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::runSortingTest(testData);
}