// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/opolin_d_simple_iteration_method/include/ops_mpi.hpp"

namespace opolin_d_simple_iteration_method_mpi {

void generateTestData(size_t size, std::vector<double> &X, std::vector<double> &A, std::vector<double> &b) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  X.resize(size);
  for (size_t i = 0; i < size; ++i) {
    X[i] = -10.0 + static_cast<double>(std::rand() % 1000) / 50.0;
  }

  A.resize(size * size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < size; ++j) {
      if (i != j) {
        A[i * size + j] = -1.0 + static_cast<double>(std::rand() % 1000) / 500.0;
        sum += std::abs(A[i * size + j]);
      }
    }
    A[i * size + i] = sum + 1.0;
  }
  b.resize(size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      b[i] += A[i * size + j] * X[j];
    }
  }
}

}  // namespace opolin_d_simple_iteration_method_mpi

TEST(opolin_d_simple_iteration_method_mpi, test_pipeline_run) {
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int size = 500;
  double epsilon = 1e-7;
  int maxIters = 10000;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> X;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  // Create data
  opolin_d_simple_iteration_method_mpi::generateTestData(size, X, A, b);
  std::vector<double> out(size, 0.0);
  // Create TaskData
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_mpi->inputs_count.emplace_back(out.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());
  auto test_task_mpi = std::make_shared<opolin_d_simple_iteration_method_mpi::TestTaskMPI>(task_data_mpi);

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
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_task_run) {
  int size = 500;
  double epsilon = 1e-7;
  int maxIters = 10000;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> X;
  // Create data
  opolin_d_simple_iteration_method_mpi::generateTestData(size, X, A, b);
  std::vector<double> out(size, 0.0);

  // Create TaskData
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_mpi->inputs_count.emplace_back(out.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());
  auto test_task_mpi = std::make_shared<opolin_d_simple_iteration_method_mpi::TestTaskMPI>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  // Create Perf attributes
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  // Create Perf analyzer
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}