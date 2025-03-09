// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/makhov_m_monte_carlo_method/include/ops_mpi.hpp"

TEST(makhov_m_monte_carlo_method_mpi, test_pipeline_run) {
  // Create data
  std::string f = "x*x";
  int numSamples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}};
  double *answerPtr = nullptr;
  double reference = 0.33;

  // Create TaskData
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
  task_data_mpi->inputs.push_back(reinterpret_cast<uint8_t *>(&numSamples));
  task_data_mpi->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_mpi->inputs_count.emplace_back(1);
  task_data_mpi->inputs_count.emplace_back(1);
  task_data_mpi->inputs_count.emplace_back(1);  // Информация о размерности интеграла
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(answerPtr));
  task_data_mpi->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_mpi = std::make_shared<makhov_m_monte_carlo_method_mpi::TestMPITaskParallel>(task_data_mpi);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto currentTimePoint = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTimePoint - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    uint8_t *answerData = task_data_mpi->outputs[0];
    double retrievedValue;
    std::memcpy(&retrievedValue, answerData, sizeof(double));
    double truncatedValue = std::round(retrievedValue * 100) / 100;
    ASSERT_EQ(reference, truncatedValue);
  }
}

TEST(makhov_m_monte_carlo_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::string f = "x*x";
  int numSamples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}};
  double *answerPtr = nullptr;
  double reference = 0.33;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(&numSamples));
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs_count.emplace_back(1);  // Информация о размерности интеграла
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(answerPtr));
    task_data_par->outputs_count.emplace_back(1);
  }

  auto test_mpi_task_parallel = std::make_shared<makhov_m_monte_carlo_method_mpi::TestMPITaskParallel>(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel->ValidationImpl());
  test_mpi_task_parallel->PreProcessingImpl();
  test_mpi_task_parallel->RunImpl();
  test_mpi_task_parallel->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    uint8_t *answerData = task_data_par->outputs[0];
    double retrievedValue;
    std::memcpy(&retrievedValue, answerData, sizeof(double));
    double truncatedValue = std::round(retrievedValue * 100) / 100;
    ASSERT_EQ(reference, truncatedValue);
  }
}
