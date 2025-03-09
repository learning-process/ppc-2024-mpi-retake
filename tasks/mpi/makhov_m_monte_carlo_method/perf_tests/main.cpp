// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/makhov_m_monte_carlo_method/include/ops_mpi.hpp"

TEST(makhov_m_monte_carlo_method_mpi, test_pipeline_run) {
  // Create data
  std::string f = "x*x";
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}};
  double *answer_ptr = nullptr;
  double reference = 0.33;

  // Create TaskData
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
  task_data_mpi->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
  task_data_mpi->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_mpi->inputs_count.emplace_back(1);
  task_data_mpi->inputs_count.emplace_back(1);
  task_data_mpi->inputs_count.emplace_back(1);  // Integral dimension info
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
  task_data_mpi->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_mpi = std::make_shared<makhov_m_monte_carlo_method_mpi::TestMPITaskParallel>(task_data_mpi);

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
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    uint8_t *answer_data = task_data_mpi->outputs[0];
    double retrieved_value = NAN;
    std::memcpy(&retrieved_value, answer_data, sizeof(double));
    double truncated_value = std::round(retrieved_value * 100) / 100;
    ASSERT_EQ(reference, truncated_value);
  }
}

TEST(makhov_m_monte_carlo_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::string f = "x*x";
  int num_samples = 1000000;
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}};
  double *answer_ptr = nullptr;
  double reference = 0.33;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(&f));
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(&num_samples));
    task_data_par->inputs.push_back(reinterpret_cast<uint8_t *>(limits.data()));
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs_count.emplace_back(1);
    task_data_par->inputs_count.emplace_back(1);  // Integral dimension info
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer_ptr));
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
    uint8_t *answer_data = task_data_par->outputs[0];
    double retrieved_value = NAN;
    std::memcpy(&retrieved_value, answer_data, sizeof(double));
    double truncated_value = std::round(retrieved_value * 100) / 100;
    ASSERT_EQ(reference, truncated_value);
  }
}
