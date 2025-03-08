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
#include "mpi/karaseva_e_num_of_alternations_signs/include/ops_mpi.hpp"

namespace {
std::vector<int> CreateRandomAlternatingSigns(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(0, 1);

  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = (dist(gen) == 0) ? -1 : 1;
  }
  return vec;
}
}  // namespace

TEST(karaseva_e_num_of_alternations_signs_mpi, test_pipeline_run) {
  constexpr int kCount = 100000000;

  // Create random data
  std::vector<int> in = CreateRandomAlternatingSigns(kCount);
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_mpi = std::make_shared<karaseva_e_num_of_alternations_signs_mpi::AlternatingSignsMPI>(task_data_mpi);

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

  // Create MPI communicator
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  // Calculate expected number of alternations for the given input
  int expected_alternations = 0;
  for (size_t i = 1; i < kCount; ++i) {
    if (in[i - 1] != in[i]) {
      ++expected_alternations;
    }
  }

  // Check if the result matches the expected alternations
  ASSERT_EQ(expected_alternations, out[0]);
}

TEST(karaseva_e_num_of_alternations_signs_mpi, test_task_run) {
  constexpr int kCount = 100000000;

  // Create random data
  std::vector<int> in = CreateRandomAlternatingSigns(kCount);
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_mpi = std::make_shared<karaseva_e_num_of_alternations_signs_mpi::AlternatingSignsMPI>(task_data_mpi);

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

  // Create MPI communicator
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  // Calculate expected number of alternations for the given input
  int expected_alternations = 0;
  for (size_t i = 1; i < kCount; ++i) {
    if (in[i - 1] != in[i]) {
      ++expected_alternations;
    }
  }

  ASSERT_EQ(expected_alternations, out[0]);
}