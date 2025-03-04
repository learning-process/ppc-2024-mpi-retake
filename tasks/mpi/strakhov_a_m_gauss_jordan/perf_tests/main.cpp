#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/strakhov_a_m_gauss_jordan/include/ops_mpi.hpp"

namespace {
std::vector<double> GenRandomVector(size_t size, int min, int max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(min, max);
  std::vector<double> random_vector(size);
  for (size_t i = 0; i < size; i++) {
    random_vector[i] = (double)(dis(gen));
  }

  return random_vector;
}
}  // namespace

TEST(strakhov_a_m_gauss_jordan_mpi, test_pipeline_run) {
  constexpr int kCount = 1000;
  boost::mpi::communicator world;
  // Create data
  std::vector<double> in = GenRandomVector(kCount * (kCount + 1), -5, 55);
  std::vector<double> ans(kCount, 0);
  if (world.rank() == 0) {
    for (size_t i = 0; i < kCount; i++) {
      ans[i] = static_cast<double>(i + 1);
    }
    for (size_t i = 0; i < kCount; i++) {
      double sum = 0;
      in[((kCount + 1) * i) + i] += (int)(in[((kCount + 1) * i) + i] == 0);
      for (size_t j = 0; j < kCount; j++) {
        sum += ans[j] * in[((kCount + 1) * i) + j];
      }
      in[((kCount + 1) * (i + 1)) - 1] = sum;
    }
  }
  std::vector<double> out(kCount, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(kCount + 1);
    task_data_mpi->inputs_count.emplace_back(kCount);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  auto test_task_mpi = std::make_shared<strakhov_a_m_gauss_jordan_mpi::TestTaskMPI>(task_data_mpi);
  // Create Task

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
    for (size_t i = 0; i < kCount; i++) {
      ASSERT_TRUE((ans[i] - out[i]) < 0.5);
    }
  }
}

TEST(strakhov_a_m_gauss_jordan_mpi, test_task_run) {
  constexpr int kCount = 1000;
  boost::mpi::communicator world;
  // Create data
  std::vector<double> in = GenRandomVector(kCount * (kCount + 1), -5, 55);
  std::vector<double> ans(kCount, 0);
  if (world.rank() == 0) {
    for (size_t i = 0; i < kCount; i++) {
      ans[i] = static_cast<double>(i + 1);
    }
    for (size_t i = 0; i < kCount; i++) {
      double sum = 0;
      in[((kCount + 1) * i) + i] += (int)(in[((kCount + 1) * i) + i] == 0);
      for (size_t j = 0; j < kCount; j++) {
        sum += ans[j] * in[((kCount + 1) * i) + j];
      }
      in[((kCount + 1) * (i + 1)) - 1] = sum;
    }
  }
  std::vector<double> out(kCount, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(kCount + 1);
    task_data_mpi->inputs_count.emplace_back(kCount);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }
  // Create Task
  auto test_task_mpi = std::make_shared<strakhov_a_m_gauss_jordan_mpi::TestTaskMPI>(task_data_mpi);

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
    for (size_t i = 0; i < kCount; i++) {
      ASSERT_TRUE((ans[i] - out[i]) < 0.5);
    }
  }
}
