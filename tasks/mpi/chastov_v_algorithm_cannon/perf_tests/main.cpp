// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/chastov_v_algorithm_cannon/include/ops_mpi.hpp"

namespace {
bool CompareMatrices(const std::vector<double> &mat1, const std::vector<double> &mat2, double epsilon = 1e-9);
}  // namespace

TEST(chastov_v_algorithm_cannon_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int k_matrix = 500;

  // Create data
  std::vector<double> matrix1;
  std::vector<double> matrix2;
  std::vector<double> result_matrix;

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix1 = std::vector<double>(k_matrix * k_matrix, 0.0);
    matrix2 = std::vector<double>(k_matrix * k_matrix, 1.0);
    result_matrix = std::vector<double>(k_matrix * k_matrix);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&k_matrix));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  // Create Task
  auto test_task_mpi = std::make_shared<chastov_v_algorithm_cannon_mpi::TestTaskMPI>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

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
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(CompareMatrices(matrix1, result_matrix));
  }
}

TEST(chastov_v_algorithm_cannon_mpi, test_task_run) {
  boost::mpi::communicator world;
  int k_matrix = 500;

  // Create data
  std::vector<double> matrix1;
  std::vector<double> matrix2;
  std::vector<double> result_matrix;

  // Create TaskData
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix1 = std::vector<double>(k_matrix * k_matrix, 0.0);
    matrix2 = std::vector<double>(k_matrix * k_matrix, 1.0);
    result_matrix = std::vector<double>(k_matrix * k_matrix, 0.0);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix1.data()));
    task_data_mpi->inputs_count.emplace_back(matrix1.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix2.data()));
    task_data_mpi->inputs_count.emplace_back(matrix2.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&k_matrix));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_matrix));
    task_data_mpi->outputs_count.emplace_back(result_matrix.size());
  }

  auto test_task_mpi = std::make_shared<chastov_v_algorithm_cannon_mpi::TestTaskMPI>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

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
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(CompareMatrices(matrix1, result_matrix));
  }
}

namespace {
bool CompareMatrices(const std::vector<double> &mat1, const std::vector<double> &mat2, double epsilon) {
  if (mat1.size() != mat2.size()) {
    return false;
  }
  for (size_t i = 0; i < mat1.size(); ++i) {
    if (std::abs(mat1[i] - mat2[i]) > epsilon) {
      return false;
    }
  }
  return true;
}
}  // namespace