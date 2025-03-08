#include <cstdint>
#include <memory>
#include <vector>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>

#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/tarakanov_d_global_opt_by_characts_two_dim_prob/include/ops_mpi.hpp"


#define RUN_PERF_TEST(testName, runMethod)                                                                    \
  TEST(tarakanov_d_global_opt_two_dim_prob_mpi, testName) {                                                   \
    boost::mpi::communicator world;                                                                           \
    double step = 0.3;                                                                                        \
    std::vector<double> area = {-10, 10, -10, 10};                                                            \
    std::vector<double> func = {5, 5};                                                                        \
    std::vector<double> constraint = {1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1,                   \
                                      2, 3, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1};                  \
    std::vector<double> out(1, 0);                                                                            \
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();               \
    if (world.rank() == 0) {                                                                                  \
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));                              \
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));                              \
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));                        \
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));                                    \
      taskDataPar->inputs_count.emplace_back(12);                                                             \
      taskDataPar->inputs_count.emplace_back(0);                                                              \
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));                              \
      taskDataPar->outputs_count.emplace_back(out.size());                                                    \
    }                                                                                                         \
    auto testClassPar = std::make_shared<tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi>(taskDataPar); \
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();                                                  \
    perfAttr->num_running = 50;                                                                               \
    const boost::mpi::timer current_timer;                                                                    \
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };                                        \
    auto perfResults = std::make_shared<ppc::core::PerfResults>();                                            \
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClassPar);                                      \
    perfAnalyzer->runMethod(perfAttr, perfResults);                                                           \
    if (world.rank() == 0) {                                                                                  \
      ppc::core::Perf::PrintPerfStatistic(perfResults);                                                       \
      EXPECT_NEAR(46.777778, out[0], 0.0001);                                                                 \
    }                                                                                                         \
  }

RUN_PERF_TEST(test_PipelineRun, PipelineRun)
RUN_PERF_TEST(test_TaskRun, TaskRun)
