#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/leontev_n_binary/include/ops_seq.hpp"

namespace {
std::vector<uint8_t> GetRandomVector(size_t rows, size_t cols) {
  std::vector<uint8_t> img(rows * cols);
  for (size_t i = 0; i < img.size(); i++) {
    img[i] = rand() % 2;
  }
  return img;
}
}  // namespace

TEST(leontev_n_binary_seq, test_pipeline_run) {
  size_t rows = 256;
  size_t cols = 256;
  std::vector<uint8_t> image = GetRandomVector(rows, cols);
  std::vector<uint32_t> actual(rows * cols);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(actual.data()));
  task_data_seq->outputs_count.emplace_back(rows);
  task_data_seq->outputs_count.emplace_back(cols);

  auto binary_segments = std::make_shared<leontev_n_binary_seq::BinarySegmentsSeq>(task_data_seq);
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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(binary_segments);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(leontev_n_binary_seq, test_task_run) {
  size_t rows = 256;
  size_t cols = 256;
  std::vector<uint8_t> image = GetRandomVector(rows, cols);
  std::vector<uint32_t> actual(rows * cols);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
  task_data_seq->inputs_count.emplace_back(rows);
  task_data_seq->inputs_count.emplace_back(cols);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(actual.data()));
  task_data_seq->outputs_count.emplace_back(rows);
  task_data_seq->outputs_count.emplace_back(cols);

  auto binary_segments = std::make_shared<leontev_n_binary_seq::BinarySegmentsSeq>(task_data_seq);
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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(binary_segments);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
