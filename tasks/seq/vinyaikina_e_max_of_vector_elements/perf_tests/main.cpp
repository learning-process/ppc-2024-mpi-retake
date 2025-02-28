
#include <gtest/gtest.h>

#include <limits>

#include "core/perf/include/perf.hpp"
#include "seq/vinyaikina_e_max_of_vector_elements/include/ops_seq.hpp"

TEST(vinyaikina_e_max_of_vector_elements_seq, test_pipeline_run) {
  const int32_t vec_size = 50000000;
  std::vector<int32_t> input_data(vec_size, 1);
  input_data[vec_size / 2] = 10;
  int32_t expected_max = 10;
  int32_t actual_max = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&actual_max));
  taskDataSeq->outputs_count.emplace_back(1);

  auto vectorMaxSequential = std::make_shared<vinyaikina_e_max_of_vector_elements_seq::VectorMaxSeq>(taskDataSeq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(vectorMaxSequential);
  perf_analyzer->pipeline_run(perf_attr, perf_results);
  ppc::core::Perf::print_perf_statistic(perf_results);

  ASSERT_EQ(expected_max, actual_max);
}

TEST(vinyaikina_e_max_of_vector_elements_seq, first_negative) {
  const int32_t count = 50000000;
  std::vector<int32_t> input_data(count, 1);
  input_data[0] = -5;
  int32_t expected_max = 1;
  int32_t actual_max = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&actual_max));
  taskDataSeq->outputs_count.emplace_back(1);

  auto vectorMaxSequential = std::make_shared<vinyaikina_e_max_of_vector_elements_seq::VectorMaxSeq>(taskDataSeq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(vectorMaxSequential);
  perf_analyzer->task_run(perf_attr, perf_results);
  ppc::core::Perf::print_perf_statistic(perf_results);

  ASSERT_EQ(expected_max, actual_max);
}
