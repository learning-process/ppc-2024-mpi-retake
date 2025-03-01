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

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi, test_pipeline_run) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N = 500000;
  std::vector<double> inputData(N);
  std::vector<double> xPar(N, 0.0);

  if (rank == 0) {
    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::generate_random_data(inputData, N);
  }

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(N);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  auto parallelRadixSort =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI>(
          taskDataPar);

  ASSERT_TRUE(parallelRadixSort->ValidationImpl()) << "Validation failed!";
  parallelRadixSort->PreProcessingImpl();
  parallelRadixSort->RunImpl();
  parallelRadixSort->PostProcessingImpl();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelRadixSort);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);

  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
  }
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi, test_task_run) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N = 500000;
  std::vector<double> inputData;
  std::vector<double> xPar(N, 0.0);

  if (rank == 0) {
    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::generate_random_data(inputData, N);
  }

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  auto parallelRadixSort =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI>(
          taskDataPar);

  ASSERT_TRUE(parallelRadixSort->ValidationImpl()) << "Validation failed!";
  parallelRadixSort->PreProcessingImpl();
  parallelRadixSort->RunImpl();
  parallelRadixSort->PostProcessingImpl();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelRadixSort);
  perfAnalyzer->TaskRun(perfAttr, perfResults);

  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
  }
}