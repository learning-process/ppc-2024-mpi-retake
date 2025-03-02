// @copyright Tarakanov Denis
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/tarakanov_d_global_opt_by_characts_two_dim_prob/include/ops_seq.hpp"

namespace {
void runPerfTest(bool isPipeline) {
    double step = 0.3;
    std::vector<double> area = {-10, 10, -10, 10};
    std::vector<double> func = {5, 5};
    std::vector<double> constraint = {1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1,
                                      2, 3, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1};
    std::vector<double> out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
        taskDataPar->inputs = {reinterpret_cast<uint8_t*>(area.data()),
                               reinterpret_cast<uint8_t*>(func.data()),
                               reinterpret_cast<uint8_t*>(constraint.data()),
                               reinterpret_cast<uint8_t*>(&step)};
        taskDataPar->inputs_count = {12, 0};
        taskDataPar->outputs = {reinterpret_cast<uint8_t*>(out.data())};
        taskDataPar->outputs_count = {static_cast<uint32_t>(out.size())};

    auto testClassPar = std::make_shared<tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential>(
        taskDataPar);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClassPar);
    
    if (isPipeline) {
        perfAnalyzer->PipelineRun(perf_attr, perfResults);
    } else {
        perfAnalyzer->TaskRun(perf_attr, perfResults);
    }
    
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    EXPECT_NEAR(215.11111, out[0], 1e-4);
}
} // namespace

TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_pipeline_run) { runPerfTest(true); }
TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_task_run) { runPerfTest(false); }
