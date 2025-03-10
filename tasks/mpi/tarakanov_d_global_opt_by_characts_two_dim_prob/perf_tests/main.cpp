#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/tarakanov_d_global_opt_by_characts_two_dim_prob/include/ops_mpi.hpp"

#define RUN_PERF_TEST(testName, runMethod)                                                                    \
  TEST(tarakanov_d_global_opt_two_dim_prob_mpi, testName) {                                                   \
    boost::mpi::communicator world;                                                                           \
    double step = 0.3;                                                                                        \
    std::vector<double> area = {-10, 10, -10, 10};                                                            \
    std::vector<double> func = {5, 5};                                                                        \
    std::vector<double> out(1, 0);                                                                            \
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();               \
    if (world.rank() == 0) {                                                                                  \
      int num_constraints = 12;                                                                               \
      std::vector<double> constraint(num_constraints * 3);                                                    \
      std::random_device rd;                                                                                  \
      std::mt19937 gen(rd());                                                                                 \
      std::uniform_real_distribution<> dis_ab(-1.0, 1.0);                                                     \
      std::uniform_real_distribution<> dis_d(0.0, 10.0);                                                      \
      double p = func[0];                                                                                     \
      double q = func[1];                                                                                     \
      for (int i = 0; i < num_constraints; i++) {                                                             \
        double a = dis_ab(gen);                                                                               \
        double b = dis_ab(gen);                                                                               \
        double d = dis_d(gen);                                                                                \
        double c = (a * p) + (b * q) + d;                                                                     \
        constraint[i * 3] = a;                                                                                \
        constraint[(i * 3) + 1] = b;                                                                          \
        constraint[(i * 3) + 2] = c;                                                                          \
      }                                                                                                       \
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));                              \
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));                              \
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));                        \
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));                                    \
      taskDataPar->inputs_count.emplace_back(num_constraints);                                                \
      taskDataPar->inputs_count.emplace_back(0);                                                              \
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));                              \
      taskDataPar->outputs_count.emplace_back(out.size());                                                    \
    }                                                                                                         \
    auto testClassPar = std::make_shared<tarakanov_d_global_opt_two_dim_prob_mpi::GlobalOptMpi>(taskDataPar); \
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();                                                  \
    perfAttr->num_running = 18;                                                                               \
    const boost::mpi::timer current_timer;                                                                    \
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };                                        \
    auto perfResults = std::make_shared<ppc::core::PerfResults>();                                            \
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClassPar);                                      \
    perfAnalyzer->runMethod(perfAttr, perfResults);                                                           \
    if (world.rank() == 0) {                                                                                  \
      ppc::core::Perf::PrintPerfStatistic(perfResults);                                                       \
    }                                                                                                         \
  }

RUN_PERF_TEST(test_PipelineRun, PipelineRun)
RUN_PERF_TEST(test_TaskRun, TaskRun)