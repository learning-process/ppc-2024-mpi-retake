#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/ersoz_b_horizontal_linear_filtering_gauss/include/ops_mpi.hpp"

TEST(ersoz_b_test_task_mpi, test_pipeline_run) {
  constexpr int kN = 512;
  std::vector<char> in(kN * kN, 0);
  for (int i = 0; i < kN; i++) {
    for (int j = 0; j < kN; j++) {
      in[(i * kN) + j] = static_cast<char>((i + j) % 256);
    }
  }

  std::vector<char> out((kN - 2) * (kN - 2), 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.push_back(in.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.push_back(out.size());

  auto task = std::make_shared<ersoz_b_test_task_mpi::TestTaskMPI>(task_data);
  ASSERT_TRUE(task->Validation());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    // Validity check
    if (perf_results) {
      ppc::core::Perf::PrintPerfStatistic(perf_results);
    } else {
      std::cerr << "Invalid performance results!" << "\n";
    }
  }
}

TEST(ersoz_b_test_task_mpi, test_task_run) {
  constexpr int kN = 512;
  std::vector<char> in(kN * kN, 0);
  for (int i = 0; i < kN; i++) {
    for (int j = 0; j < kN; j++) {
      in[(i * kN) + j] = static_cast<char>((i + j) % 256);
    }
  }

  std::vector<char> out((kN - 2) * (kN - 2), 0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.push_back(in.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.push_back(out.size());

  auto task = std::make_shared<ersoz_b_test_task_mpi::TestTaskMPI>(task_data);
  ASSERT_TRUE(task->Validation());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    if (perf_results) {
      ppc::core::Perf::PrintPerfStatistic(perf_results);
    } else {
      std::cerr << "Invalid performance results!" << "\n";
    }
  }
}
