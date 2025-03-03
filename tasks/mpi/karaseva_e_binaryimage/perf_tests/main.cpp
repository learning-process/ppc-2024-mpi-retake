#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/karaseva_e_binaryimage/include/ops_mpi.hpp"

TEST(karaseva_e_binaryimage_mpi, test_pipeline_run) {
  const int rows = 1250;
  const int columns = 1250;
  std::vector<int> in(rows * columns);
  std::vector<int> out(rows * columns, -1);
  std::vector<int> expected_out(rows * columns);

  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < columns; y++) {
      int pos = (x * columns) + y;  // Added parentheses to specify order of operations
      if (x < 50) {
        in[pos] = 0;
        expected_out[pos] = 2;
      } else if (x == 50 || x == 52) {
        in[pos] = 1;
        expected_out[pos] = 1;
      } else if (x == 51) {
        in[pos] = 0;
        expected_out[pos] = 3;
      } else {
        in[pos] = 0;
        expected_out[pos] = 4;
      }
    }
  }

  // Creating TaskData
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(rows);
  task_data_mpi->inputs_count.emplace_back(columns);
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(rows);
  task_data_mpi->outputs_count.emplace_back(columns);

  // Creating the task
  auto test_task_mpi = std::make_shared<karaseva_e_binaryimage_mpi::TestMPITaskParallel>(task_data_mpi);

  // Performance setup
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Running performance analysis
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (std::size_t i = 0; i < expected_out.size(); i++) {
    if (expected_out[i] != out[i]) {
      std::cout << "Mismatch at index " << i << ": expected " << expected_out[i] << ", got " << out[i] << "\n";
    }
  }

  ASSERT_EQ(expected_out, out);
}

TEST(karaseva_e_binaryimage_mpi, test_task_run) {
  const int rows = 1250;
  const int columns = 1250;
  std::vector<int> in(rows * columns);
  std::vector<int> out(rows * columns, -1);
  std::vector<int> expected_out(rows * columns);

  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < columns; y++) {
      int pos = (x * columns) + y;  // Added parentheses to specify order of operations
      if (x < 50) {
        in[pos] = 0;
        expected_out[pos] = 2;
      } else if (x == 50 || x == 52) {
        in[pos] = 1;
        expected_out[pos] = 1;
      } else if (x == 51) {
        in[pos] = 0;
        expected_out[pos] = 3;
      } else {
        in[pos] = 0;
        expected_out[pos] = 4;
      }
    }
  }

  // Creating TaskData
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(rows);
  task_data_mpi->inputs_count.emplace_back(columns);
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(rows);
  task_data_mpi->outputs_count.emplace_back(columns);

  // Creating the task
  auto test_task_mpi = std::make_shared<karaseva_e_binaryimage_mpi::TestMPITaskParallel>(task_data_mpi);

  // Performance setup
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Running performance analysis
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(expected_out, out);
}