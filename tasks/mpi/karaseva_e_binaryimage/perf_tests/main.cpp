#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/karaseva_e_binaryimage/include/ops_mpi.hpp"

namespace {

// Function to generate a random binary image of given size
std::vector<uint8_t> createRandomBinaryImg(size_t rows, size_t cols) {
  std::vector<uint8_t> img(rows * cols);
  for (auto &px : img) {
    px = rand() % 2;
  }
  return img;
}

}  // namespace

// Test for the pipeline run
TEST(karaseva_e_binaryimage_mpi, test_pipeline_run) {
  constexpr int kRows = 100;
  constexpr int kCols = 100;

  // Create binary image data
  std::vector<uint8_t> image = createRandomBinaryImg(kRows, kCols);
  std::vector<int> output(kRows * kCols, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_mpi->inputs_count.emplace_back(kRows);
  task_data_mpi->inputs_count.emplace_back(kCols);
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_mpi->outputs_count.emplace_back(kRows);
  task_data_mpi->outputs_count.emplace_back(kCols);

  // Create Task
  auto test_task_mpi = std::make_shared<karaseva_e_binaryimage_mpi::TestMPITaskParallel>(task_data_mpi);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and initialize perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer and run the pipeline
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  // Create Perf analyzer for MPI
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

// Test for the task run
TEST(karaseva_e_binaryimage_mpi, test_task_run) {
  constexpr int kRows = 100;
  constexpr int kCols = 100;

  // Create binary image data
  std::vector<uint8_t> image = createRandomBinaryImg(kRows, kCols);
  std::vector<int> output(kRows * kCols, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data_mpi->inputs_count.emplace_back(kRows);
  task_data_mpi->inputs_count.emplace_back(kCols);
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_mpi->outputs_count.emplace_back(kRows);
  task_data_mpi->outputs_count.emplace_back(kCols);

  // Create Task
  auto test_task_mpi = std::make_shared<karaseva_e_binaryimage_mpi::TestMPITaskParallel>(task_data_mpi);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and initialize perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer and run the task
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  // Create Perf analyzer for MPI
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}