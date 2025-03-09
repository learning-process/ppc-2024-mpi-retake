#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_pipeline_run) {
  double a = -1.0;
  double b = 4.0;
  double result = 0;
  std::function<double(double *)> f = [](const double *x) { return ((*x) * (*x)) + (2 * (*x)) + 1; };

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_mpi->inputs_count.emplace_back(2);
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  task_data_mpi->outputs_count.emplace_back(1);

  auto test_task_mpi =
      std::make_shared<prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel>(task_data_mpi, f);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);

    double expected_result = -1.0;
    double tolerance = 1e-6;
    EXPECT_NEAR(result, expected_result, tolerance);
  }
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_task_run) {
  double a = -1.0;
  double b = 4.0;
  double result = 0;
  std::function<double(double *)> f = [](const double *x) { return ((*x) * (*x)) + (2 * (*x)) + 1; };

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_mpi->inputs_count.emplace_back(2);
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  task_data_mpi->outputs_count.emplace_back(1);

  auto test_task_mpi =
      std::make_shared<prokhorov_n_global_search_algorithm_strongin_mpi::TestTaskParallel>(task_data_mpi, f);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);

    double expected_result = -1.0;
    double tolerance = 1e-6;
    EXPECT_NEAR(result, expected_result, tolerance);
  }
}