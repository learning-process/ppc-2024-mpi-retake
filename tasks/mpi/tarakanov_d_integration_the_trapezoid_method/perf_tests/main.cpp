// Copyright 2025 Tarakanov Denis
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/tarakanov_d_integration_the_trapezoid_method/include/ops_mpi.hpp"

using namespace tarakanov_d_integration_the_trapezoid_method_mpi;

#define MY_TEST(test_name, test_function)                                                                    \
  TEST(tarakanov_d_trapezoid_method_mpi, test_name) {                                                        \
    double a = 0.0;                                                                                          \
    double b = 1.0;                                                                                          \
    double h = 0.000000005;                                                                                  \
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
    auto task = std::make_shared<IntegrationTheTrapezoidMethodMPI>(taskData);                                \
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
    boost::mpi::communicator world;                                                                          \
    if (world.rank() == 0) {                                                                                 \
      ppc::core::Perf::PrintPerfStatistic(perfResults);                                                      \
      EXPECT_NEAR(out, 0.25, 1e-3);                                                                          \
    }                                                                                                        \
  }

MY_TEST(test_pipeline_run, PipelineRun);
MY_TEST(test_task_run, TaskRun);