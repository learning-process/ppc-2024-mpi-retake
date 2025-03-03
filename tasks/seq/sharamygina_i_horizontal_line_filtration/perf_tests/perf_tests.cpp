#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sharamygina_i_horizontal_line_filtration/include/ops_seq.h"

namespace sharamygina_i_horizontal_line_filtration_seq {
namespace {
std::vector<unsigned int> GetImage(int k_rows, int k_cols) {
  std::vector<unsigned int> temporary_im(k_rows * k_cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, std::numeric_limits<unsigned int>::max());

  for (int i = 0; i < k_rows; i++) {
    for (int j = 0; j < k_cols; j++) {
      temporary_im[(i * k_cols) + j] = dist(gen);
    }
  }
  return temporary_im;
}
}  // namespace
}  // namespace sharamygina_i_horizontal_line_filtration_seq

TEST(sharamygina_i_horizontal_line_filtration_seq, test_pipeline_run) {
  constexpr int kRows = 6000;
  constexpr int kCols = 6000;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<unsigned int> input = sharamygina_i_horizontal_line_filtration_seq::GetImage(kRows, kCols);
  std::vector<unsigned int> output(kRows * kCols);

  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(kRows);
  task_data->inputs_count.push_back(kCols);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.push_back(output.size());

  auto test_task =
      std::make_shared<sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1;

  const auto start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(test_task);

  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(sharamygina_i_horizontal_line_filtration_seq, test_task_run) {
  constexpr int kRows = 6000;
  constexpr int kCols = 6000;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<unsigned int> input = sharamygina_i_horizontal_line_filtration_seq::GetImage(kRows, kCols);
  std::vector<unsigned int> output(kRows * kCols);

  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(kRows);
  task_data->inputs_count.push_back(kCols);
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.push_back(output.size());

  auto test_task =
      std::make_shared<sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1;

  const auto start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(test_task);

  perf->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
