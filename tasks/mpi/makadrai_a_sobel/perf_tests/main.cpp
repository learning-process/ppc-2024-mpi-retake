#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/makadrai_a_sobel/include/ops_mpi.hpp"

TEST(makadrai_a_sobel_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int height_img = 2500;
  int width_img = 2500;
  std::vector<int> img;
  std::vector<int> res;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img.resize(width_img * height_img);
    res.resize(width_img * height_img);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

    task_data->inputs_count.emplace_back(width_img);
    task_data->inputs_count.emplace_back(height_img);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(width_img);
    task_data->outputs_count.emplace_back(height_img);
  }

  // Create Task
  auto test_task_mpi = std::make_shared<makadrai_a_sobel_mpi::Sobel>(task_data);

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

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(makadrai_a_sobel_mpi, test_task_run) {
  boost::mpi::communicator world;
  int height_img = 2500;
  int width_img = 2500;
  std::vector<int> img;
  std::vector<int> res;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    img.resize(width_img * height_img);
    res.resize(width_img * height_img);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));

    task_data->inputs_count.emplace_back(width_img);
    task_data->inputs_count.emplace_back(height_img);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(width_img);
    task_data->outputs_count.emplace_back(height_img);
  }
  // Create Task
  auto test_task_mpi = std::make_shared<makadrai_a_sobel_mpi::Sobel>(task_data);

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
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}
