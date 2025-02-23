#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kavtorev_d_radix_double_sort/include/ops_mpi.hpp"

namespace mpi = boost::mpi;
using namespace kavtorev_d_radix_double_sort;

TEST(kavtorev_d_radix_double_sort_mpi, test_pipeline_run) {
  mpi::environment env;
  mpi::communicator world;

  int N = 10000000;

  std::vector<double> inputData;
  if (world.rank() == 0) {
    inputData.resize(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);

    for (int i = 0; i < N; ++i) {
      inputData[i] = dist(gen);
    }
  }

  std::vector<double> xPar(N, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    task_data_mpi->inputs_count.emplace_back(N);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    task_data_mpi->outputs_count.emplace_back(N);
  }

  auto test_task_mpi = std::make_shared<kavtorev_d_radix_double_sort::RadixSortParallel>(task_data_mpi);
  ASSERT_TRUE(test_task_mpi->ValidationImpl());
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
  }
}

TEST(kavtorev_d_radix_double_sort_mpi, test_task_run) {
  mpi::environment env;
  mpi::communicator world;

  int N = 1000000;
  std::vector<double> inputData;
  if (world.rank() == 0) {
    inputData.resize(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);

    for (int i = 0; i < N; ++i) {
      inputData[i] = dist(gen);
    }
  }

  std::vector<double> xPar(N, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    task_data_mpi->inputs_count.emplace_back(N);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    task_data_mpi->outputs_count.emplace_back(N);
  }

  auto test_task_mpi = std::make_shared<kavtorev_d_radix_double_sort::RadixSortParallel>(task_data_mpi);
  ASSERT_TRUE(test_task_mpi->ValidationImpl());
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
  }
}