// Copyright 2024 Kabalova Valeria
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kabalova_v_strongin/include/strongin.h"

TEST(kabalova_v_strongin_seq, test_pipeline_run) {
  double left = -1.0;
  double right = 7.0;
  double res1 = 0;
  double res2 = 0;
  std::function<double(double)> f = [](double x) { return std::exp(x) - 1.0; };
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  auto test_task_sequential = std::make_shared<kabalova_v_strongin_seq::TestTaskSequential>(task_data_seq, f);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 30;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(kabalova_v_strongin_seq, test_task_run) {
  double left = -1.0;
  double right = 7.0;
  double res1 = 0;
  double res2 = 0;
  std::function<double(double)> f = [](double x) { return std::exp(x) - 1.0; };
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res1));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res2));
  task_data_seq->outputs_count.emplace_back(2);

  // Create Task
  auto test_task_sequential = std::make_shared<kabalova_v_strongin_seq::TestTaskSequential>(task_data_seq, f);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 30;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}