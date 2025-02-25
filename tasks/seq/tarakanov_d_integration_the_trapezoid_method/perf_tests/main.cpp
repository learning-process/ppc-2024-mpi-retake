// Copyright 2025 Tarakanov Denis
#include <gtest/gtest.h>

#include <boost/math/quadrature/trapezoidal.hpp>
#include <iostream>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/tarakanov_d_integration_the_trapezoid_method/include/ops_seq.hpp"

using namespace tarakanov_d_integration_the_trapezoid_method_seq;

#define my_test(test_name, test_function)                                                                    \
  TEST(trapezoid_method_perf_test, test_name) {                                                              \
    double a = 0.0;                                                                                          \
    double b = 1.0;                                                                                          \
    double h = 0.1;                                                                                          \
                                                                                                             \
    auto taskData = std::make_shared<ppc::core::TaskData>();                                                 \
    taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));                                             \
    taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));                                             \
    taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&h));                                             \
    taskData->inputs_count.push_back(3);                                                                     \
                                                                                                             \
    double out = 0.0;                                                                                        \
    taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&out));                                          \
    taskData->outputs_count.push_back(1);                                                                    \
                                                                                                             \
    auto task = std::make_shared<IntegrationTheTrapezoidMethodSequential>(taskData);                         \
                                                                                                             \
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();                                                 \
    perfAttr->num_running = 10;                                                                              \
    const auto t0 = std::chrono::high_resolution_clock::now();                                               \
    perfAttr->current_timer = [&] {                                                                          \
      auto current_time_point = std::chrono::high_resolution_clock::now();                                   \
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count(); \
      return static_cast<double>(duration) * 1e-9;                                                           \
    };                                                                                                       \
                                                                                                             \
    auto perfResults = std::make_shared<ppc::core::PerfResults>();                                           \
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);                                             \
    perfAnalyzer->test_function(perfAttr, perfResults);                                                      \
    ppc::core::Perf::PrintPerfStatistic(perfResults);                                                        \
                                                                                                             \
    double expected_result = boost::math::quadrature::trapezoidal([](double x) { return x / 2; }, a, b);     \
    EXPECT_DOUBLE_EQ(out, expected_result);                                                                  \
  }

my_test(test_pipeline_run, PipelineRun);
my_test(test_task_run, TaskRun);
