#include <gtest/gtest.h>

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/strakhov_a_fox_algorithm/include/ops_mpi.hpp"

namespace {
std::vector<double> MultiplyMatrices(std::vector<double>& a, std::vector<double>& b, size_t n) {
  std::vector<double> c(a.size(), 0);
  for (unsigned int i = 0; i < n; ++i) {
    for (unsigned int j = 0; j < n; ++j) {
      for (unsigned int k = 0; k < n; ++k) {
        c[(i * n) + j] += (a[(i * n) + k] * b[(k * n) + j]);
      }
    }
  }
  return c;
}

std::vector<double> CreateRandomVal(double min_v, double max_v, size_t s) {
  std::random_device dev;
  std::mt19937 random(dev());
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(min_v, max_v);
  std::vector<double> ans(s, 0);
  for (size_t i = 0; i < s; i++) {
    ans[i] = dis(gen);
  }
  return ans;
}
}  // namespace

TEST(strakhov_a_fox_algorithm_mpi, test_pipeline_run) {
  constexpr size_t kCount = 200;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> a(kCount * kCount, 0);
  std::vector<double> b(kCount * kCount, 0);
  std::vector<double> ans(kCount * kCount, 0);
  if (world.rank() == 0) {
    a = CreateRandomVal(-100, 100, kCount * kCount);
    b = CreateRandomVal(-100, 100, kCount * kCount);
    ans = MultiplyMatrices(a, b, kCount);
  }
  std::vector<double> out(kCount * kCount, 0);
  broadcast(world, a, 0);
  broadcast(world, b, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_mpi->inputs_count.emplace_back(kCount);
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_mpi = std::make_shared<strakhov_a_fox_algorithm_mpi::TestTaskMPI>(task_data_mpi);

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
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
  if (world.rank() == 0) {
    for (size_t i = 0; i < out.size(); i++) {
      ASSERT_FLOAT_EQ(ans[i], out[i]);
    }
  }
}

TEST(strakhov_a_fox_algorithm_mpi, test_task_run) {
  constexpr size_t kCount = 200;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> a(kCount * kCount, 0);
  std::vector<double> b(kCount * kCount, 0);
  std::vector<double> ans(kCount * kCount, 0);
  if (world.rank() == 0) {
    a = CreateRandomVal(-100, 100, kCount * kCount);
    b = CreateRandomVal(-100, 100, kCount * kCount);
    ans = MultiplyMatrices(a, b, kCount);
  }
  std::vector<double> out(kCount * kCount, 0);
  broadcast(world, a, 0);
  broadcast(world, b, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_mpi->inputs_count.emplace_back(kCount);
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());
  // Create Task
  auto test_task_mpi = std::make_shared<strakhov_a_fox_algorithm_mpi::TestTaskMPI>(task_data_mpi);

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
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  if (world.rank() == 0) {
    for (size_t i = 0; i < out.size(); i++) {
      ASSERT_FLOAT_EQ(ans[i], out[i]);
    }
  }
}
