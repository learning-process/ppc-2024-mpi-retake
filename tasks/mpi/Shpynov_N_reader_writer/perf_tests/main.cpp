#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/Shpynov_N_reader_writer/include/readers_writers_mpi.hpp"

TEST(shpynov_n_readers_writers_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  constexpr int kCount = 500;

  std::vector<int> crit_res(kCount, 5);
  std::vector<int> returned_result(kCount, 0);
  std::vector<int> expected_result = crit_res;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int writers_count = 0;
    for (int i = 1; i < world.size(); i++) {
      if (i % 2 != 0) {
        writers_count++;
      }
    }
    for (unsigned long i = 0; i < crit_res.size(); i++) {
      expected_result[i] = writers_count + crit_res[i];
    }
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(crit_res.data()));
    task_data_mpi->inputs_count.emplace_back(crit_res.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
    task_data_mpi->outputs_count.emplace_back(crit_res.size());
  }
  auto test_task_mpi = std::make_shared<shpynov_n_readers_writers_mpi::TestTaskMPI>(task_data_mpi);
  shpynov_n_readers_writers_mpi::TestTaskMPI test_task_mpi_1(task_data_mpi);
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

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(expected_result, returned_result);
  }
}

TEST(shpynov_n_readers_writers_mpi, test_task_run) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  constexpr int kCount = 500;

  std::vector<int> crit_res(kCount, 5);
  std::vector<int> returned_result(kCount, 0);
  std::vector<int> expected_result = crit_res;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int writers_count = 0;
    for (int i = 1; i < world.size(); i++) {
      if (i % 2 != 0) {
        writers_count++;
      }
    }
    for (unsigned long i = 0; i < crit_res.size(); i++) {
      expected_result[i] = writers_count + crit_res[i];
    }
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(crit_res.data()));
    task_data_mpi->inputs_count.emplace_back(crit_res.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
    task_data_mpi->outputs_count.emplace_back(crit_res.size());
  }
  auto test_task_mpi = std::make_shared<shpynov_n_readers_writers_mpi::TestTaskMPI>(task_data_mpi);
  shpynov_n_readers_writers_mpi::TestTaskMPI test_task_mpi_1(task_data_mpi);
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
  perf_analyzer->TaskRun(perf_attr, perf_results);
  // Create Perf analyzer

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(expected_result, returned_result);
  }
}
