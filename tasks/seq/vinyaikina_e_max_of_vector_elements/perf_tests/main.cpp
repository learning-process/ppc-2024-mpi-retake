
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/vinyaikina_e_max_of_vector_elements/include/ops_seq.hpp"

TEST(vinyaikina_e_max_of_vector_elements_seq, test_pipeline_run) {
  const int32_t vec_size = 50000000;
  std::vector<int32_t> input_data(vec_size, 1);
  input_data[vec_size / 2] = 10;
  int32_t expected_max = 10;
  int32_t actual_max = std::numeric_limits<int32_t>::min();

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(input_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&actual_max));
  task_data_seq->outputs_count.emplace_back(1);

  auto vector_max_sequential = std::make_shared<vinyaikina_e_max_of_vector_elements_seq::VectorMaxSeq>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(vector_max_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(expected_max, actual_max);
}

TEST(vinyaikina_e_max_of_vector_elements_seq, first_negative) {
  const int32_t count = 50000000;
  std::vector<int32_t> input_data(count, 1);
  input_data[0] = -5;
  int32_t expected_max = 1;
  int32_t actual_max = std::numeric_limits<int32_t>::min();

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data_seq->inputs_count.emplace_back(input_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&actual_max));
  task_data_seq->outputs_count.emplace_back(1);

  auto vector_max_sequential = std::make_shared<vinyaikina_e_max_of_vector_elements_seq::VectorMaxSeq>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(vector_max_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(expected_max, actual_max);
}
