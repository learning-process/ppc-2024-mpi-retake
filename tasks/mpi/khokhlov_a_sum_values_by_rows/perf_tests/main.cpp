#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/khokhlov_a_sum_values_by_rows/include/ops_mpi.hpp"

TEST(khokhlov_a_sum_values_by_rows_mpi, test_pipeline_Run) {
  boost::mpi::communicator world;

  int cols = 12000;
  int rows = 12000;

  // Create data
  std::vector<int> in(cols * rows, 0);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      in[i] += i * cols + j;
    }
  }
  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < cols; j++) {
      tmp_sum += in[(i * cols) + j];
    }
    expect[i] += tmp_sum;
  }
  std::vector<int> out(rows, 0);
  // Create TaskData
  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count.emplace_back(in.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_par->outputs_count.emplace_back(out.size());
  }

  auto test_task_par = std::make_shared<khokhlov_a_sum_values_by_rows_mpi::SumValByRowsMpi>(task_data_par);
  ASSERT_EQ(test_task_par->ValidationImpl(), true);
  test_task_par->PreProcessingImpl();
  test_task_par->RunImpl();
  test_task_par->PostProcessingImpl();

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

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_par);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(expect, out);
  }
}

TEST(khokhlov_a_sum_values_by_rows_mpi, test_task_Run) {
  boost::mpi::communicator world;

  int cols = 12000;
  int rows = 12000;

  // Create data
  std::vector<int> in(cols * rows, 0);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      in[i] += i * cols + j;
    }
  }
  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < cols; j++) {
      tmp_sum += in[(i * cols) + j];
    }
    expect[i] += tmp_sum;
  }
  std::vector<int> out(rows, 0);
  // Create TaskData
  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count.emplace_back(in.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_par->outputs_count.emplace_back(out.size());
  }

  auto test_task_par = std::make_shared<khokhlov_a_sum_values_by_rows_mpi::SumValByRowsMpi>(task_data_par);
  ASSERT_EQ(test_task_par->ValidationImpl(), true);
  test_task_par->PreProcessingImpl();
  test_task_par->RunImpl();
  test_task_par->PostProcessingImpl();

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_par);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(expect, out);
  }
}
