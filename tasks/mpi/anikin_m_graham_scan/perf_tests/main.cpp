// Anikin Maksim 2025
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/anikin_m_graham_scan/include/ops_mpi.hpp"

namespace {
void CreateRandomData(std::vector<anikin_m_graham_scan_mpi::Pt> &alg_in, int count) {
  alg_in.clear();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 100.0);
  anikin_m_graham_scan_mpi::Pt rand;
  for (int i = 0; i < count; i++) {
    rand.x = (int)dis(gen);
    rand.y = (int)dis(gen);
    alg_in.push_back(rand);
  }
}
}  // namespace

TEST(anikin_m_graham_scan_mpi, test_pipeline_run) {
  constexpr int kCount = 2000000;

  std::vector<anikin_m_graham_scan_mpi::Pt> in;
  std::vector<anikin_m_graham_scan_mpi::Pt> out;

  CreateRandomData(in, kCount);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());

  // Create Task
  auto test_task_mpi = std::make_shared<anikin_m_graham_scan_mpi::TestTaskMPI>(task_data_mpi);

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

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
  ASSERT_EQ(true, true);
}

TEST(anikin_m_graham_scan_mpi, test_task_run) {
  constexpr int kCount = 2000000;

  std::vector<anikin_m_graham_scan_mpi::Pt> in;
  std::vector<anikin_m_graham_scan_mpi::Pt> out;

  CreateRandomData(in, kCount);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());

  // Create Task
  auto test_task_mpi = std::make_shared<anikin_m_graham_scan_mpi::TestTaskMPI>(task_data_mpi);

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
  // Create Perf analyzer
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
  ASSERT_EQ(true, true);
}