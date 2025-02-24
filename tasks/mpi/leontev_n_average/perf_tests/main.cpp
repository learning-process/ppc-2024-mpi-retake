// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/leontev_n_average/include/ops_mpi.hpp"

inline void taskEmplacement(std::shared_ptr<ppc::core::TaskData>& taskDataPar, std::vector<int>& global_vec,
                            std::vector<int32_t>& global_avg) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
  taskDataPar->inputs_count.emplace_back(global_vec.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_avg.data()));
  taskDataPar->outputs_count.emplace_back(global_avg.size());
}

TEST(leontev_n_average_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_avg(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 30000000;
    global_vec = std::vector<int>(count_size_vector, 1);
    taskEmplacement(taskDataPar, global_vec, global_avg);
  }
  auto MPIVecAvgParallel = std::make_shared<leontev_n_average_mpi::MPIVecAvgParallel>(taskDataPar);
  ASSERT_EQ(MPIVecAvgParallel->ValidationImpl(), true);
  MPIVecAvgParallel->PreProcessingImpl();
  MPIVecAvgParallel->RunImpl();
  MPIVecAvgParallel->PostProcessingImpl();
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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(MPIVecAvgParallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(global_avg[0], 1);
  }
}

TEST(leontev_n_average_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_avg(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 30000000;
    global_vec = std::vector<int>(count_size_vector, 1);
    taskEmplacement(taskDataPar, global_vec, global_avg);
  }
  auto MPIVecAvgParallel = std::make_shared<leontev_n_average_mpi::MPIVecAvgParallel>(taskDataPar);
  ASSERT_EQ(MPIVecAvgParallel->ValidationImpl(), true);
  MPIVecAvgParallel->PreProcessingImpl();
  MPIVecAvgParallel->RunImpl();
  MPIVecAvgParallel->PostProcessingImpl();
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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(MPIVecAvgParallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(global_avg[0], 1);
  }
}
