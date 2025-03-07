#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/dudchenko_o_sleeping_barber/include/ops_seq.hpp"

TEST(dudchenko_o_sleeping_barber_seq, test_pipeline_run) {
  const int seats = 3;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(seats);
  task_data_seq->outputs_count.emplace_back(sizeof(int));

  auto output_value = std::make_shared<std::vector<uint8_t>>(sizeof(int));
  task_data_seq->outputs.push_back(output_value->data());

  auto test_task_sequential = std::make_shared<dudchenko_o_sleeping_barber_seq::TestSleepingBarber>(task_data_seq);

  ASSERT_TRUE(test_task_sequential->ValidationImpl());
  ASSERT_TRUE(test_task_sequential->PreProcessingImpl());
  ASSERT_TRUE(test_task_sequential->RunImpl());
  ASSERT_TRUE(test_task_sequential->PostProcessingImpl());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ASSERT_TRUE(perf_results != nullptr);
}

TEST(dudchenko_o_sleeping_barber_seq, test_task_run) {
  const int seats = 3;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(seats);
  task_data_seq->outputs_count.emplace_back(sizeof(int));

  auto output_value = std::make_shared<std::vector<uint8_t>>(sizeof(int));
  task_data_seq->outputs.push_back(output_value->data());

  auto test_task_sequential = std::make_shared<dudchenko_o_sleeping_barber_seq::TestSleepingBarber>(task_data_seq);

  ASSERT_TRUE(test_task_sequential->ValidationImpl());
  ASSERT_TRUE(test_task_sequential->PreProcessingImpl());
  ASSERT_TRUE(test_task_sequential->RunImpl());
  ASSERT_TRUE(test_task_sequential->PostProcessingImpl());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  ASSERT_TRUE(perf_results != nullptr);
}
