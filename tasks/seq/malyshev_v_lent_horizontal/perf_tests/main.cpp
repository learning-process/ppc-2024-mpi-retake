#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/malyshev_v_lent_horizontal/include/ops_seq.hpp"

TEST(malyshev_v_lent_horizontal_seq, Pipeline_Run) {
  const size_t rows = 1000;
  const size_t cols = 1000;

  auto matrix = malyshev_v_lent_horizontal_seq::GetRandomMatrix(rows, cols);
  auto vector = malyshev_v_lent_horizontal_seq::GetRandomVector(cols);
  std::vector<double> result(rows, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(cols);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(rows);

  auto task = std::make_shared<malyshev_v_lent_horizontal_seq::MatrixVectorMultiplication>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(malyshev_v_lent_horizontal_seq, Task_Run) {
  const size_t rows = 1000;
  const size_t cols = 1000;

  auto matrix = malyshev_v_lent_horizontal_seq::GetRandomMatrix(rows, cols);
  auto vector = malyshev_v_lent_horizontal_seq::GetRandomVector(cols);
  std::vector<double> result(rows, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  task_data->inputs_count.emplace_back(rows);
  task_data->inputs_count.emplace_back(cols);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(rows);

  auto task = std::make_shared<malyshev_v_lent_horizontal_seq::MatrixVectorMultiplication>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}