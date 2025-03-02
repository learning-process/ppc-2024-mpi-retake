#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

namespace mpi = boost::mpi;
using namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi;

double computeValue(int i) { return std::sin(i) * 1e9 + std::cos(i * 0.5) * 1e8; }

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi_perf, test_pipeline_run) {
  mpi::environment env;
  mpi::communicator world;

  int size = 10000000;
  std::vector<double> in;
  std::vector<double> out(size, 0.0);

  if (world.rank() == 0) {
    in.resize(size);
    for (int i = 0; i < size; ++i) {
      in[i] = computeValue(i);
    }
  }

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs = {reinterpret_cast<uint8_t *>(&size), reinterpret_cast<uint8_t *>(in.data())};
    task_data_mpi->inputs_count = {1, static_cast<unsigned int>(size)};
    task_data_mpi->outputs = {reinterpret_cast<uint8_t *>(out.data())};
    task_data_mpi->outputs_count = {static_cast<unsigned int>(size)};
  }

  auto sortTask = std::make_shared<TestTaskMPI>(task_data_mpi);
  ASSERT_TRUE(sortTask->ValidationImpl());
  sortTask->PreProcessingImpl();
  sortTask->RunImpl();
  sortTask->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sortTask);
  perfAnalyzer->PipelineRun(perf_attr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi_perf, test_task_run) {
  mpi::environment env;
  mpi::communicator world;

  int size = 10000000;
  std::vector<double> in;
  std::vector<double> out(size, 0.0);

  if (world.rank() == 0) {
    in.resize(size);
    for (int i = 0; i < size; ++i) {
      in[i] = computeValue(i);
    }
  }

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

   if (world.rank() == 0) {
    task_data_mpi->inputs = {reinterpret_cast<uint8_t *>(&size), reinterpret_cast<uint8_t *>(in.data())};
    task_data_mpi->inputs_count = {1, static_cast<unsigned int>(size)};
    task_data_mpi->outputs = {reinterpret_cast<uint8_t *>(out.data())};
    task_data_mpi->outputs_count = {static_cast<unsigned int>(size)};
  }

  auto sortTask = std::make_shared<TestTaskMPI>(task_data_mpi);
  ASSERT_TRUE(sortTask->ValidationImpl());
  sortTask->PreProcessingImpl();
  sortTask->RunImpl();
  sortTask->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sortTask);
  perfAnalyzer->TaskRun(perf_attr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
  }
}