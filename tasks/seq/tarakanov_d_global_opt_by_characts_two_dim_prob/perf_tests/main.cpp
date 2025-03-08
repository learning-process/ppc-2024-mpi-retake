// @copyright Tarakanov Denis
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/tarakanov_d_global_opt_by_characts_two_dim_prob/include/ops_seq.hpp"

namespace {
void RunPerfTest(bool is_pipeline) {
  double step = 0.3;
  std::vector<double> area = {-10, 10, -10, 10};
  std::vector<double> func = {5, 5};
  std::vector<double> constraint = {1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1,
                                    2, 3, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1};
  std::vector<double> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs = {reinterpret_cast<uint8_t*>(area.data()), reinterpret_cast<uint8_t*>(func.data()),
                           reinterpret_cast<uint8_t*>(constraint.data()), reinterpret_cast<uint8_t*>(&step)};
  task_data_seq->inputs_count = {12, 0};
  task_data_seq->outputs = {reinterpret_cast<uint8_t*>(out.data())};
  task_data_seq->outputs_count = {static_cast<uint32_t>(out.size())};

  auto test_class_par = std::make_shared<tarakanov_d_global_opt_two_dim_prob_seq::GlobalOptSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_class_par);

  if (is_pipeline) {
    perf_analyzer->PipelineRun(perf_attr, perf_results);
  } else {
    perf_analyzer->TaskRun(perf_attr, perf_results);
  }

  ppc::core::Perf::PrintPerfStatistic(perf_results);
  EXPECT_NEAR(215.11111, out[0], 1e-4);
}
}  // namespace

TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_pipeline_run) { RunPerfTest(true); }
TEST(tarakanov_d_global_opt_two_dim_prob_seq, test_task_run) { RunPerfTest(false); }
