#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <mpi.h>
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

namespace {

void GenerateComputedData(std::vector<double> &data, int s, double min = -1e9, double max = 1e9) {
  data.resize(s);

  for (int i = 0; i < s; ++i) {
    data[i] = min + (max - min) * (std::sin(i * 0.1) + 1) / 2;
  }
}

}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, test_pipeline_run) {
  int world_rank_ = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int s = 1000000;
  std::vector<double> in(s);
  std::vector<double> out(s, 0.0);

  if (world_rank_ == 0) {GenerateComputedData(in, s);
  }

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(s);
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(s);

  // Create Task
  auto test_task_mpi =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI>(task_data_mpi);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world_rank_ == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, test_task_run) {
  int world_rank_ = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int s = 1000000;
  std::vector<double> in(s);
  std::vector<double> out(s, 0.0);

  if (world_rank_ == 0) {GenerateComputedData(in, s);
  }

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(s);
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(s);

  // Create Task
  auto test_task_mpi =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskMPI>(task_data_mpi);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world_rank_ == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}