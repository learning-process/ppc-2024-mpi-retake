// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kalinin_d_vector_dot_product/include/ops_seq.hpp"

namespace {
int offset = 0;
}  // namespace

namespace {
std::vector<int> CreateRandomVector(int v_size) {
  std::vector<int> vec(v_size);
  std::mt19937 gen;
  gen.seed((unsigned)time(nullptr) + ++offset);
  for (int i = 0; i < v_size; i++) {
    vec[i] = static_cast<int>(gen() % 100);
  }
  return vec;
}
}  // namespace

TEST(kalinin_d_vector_dot_product_seq, test_pipeline_run) {
  const int count = 42000000;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = CreateRandomVector(count);
  std::vector<int> v2 = CreateRandomVector(count);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  task_data_seq->inputs_count.emplace_back(v1.size());
  task_data_seq->inputs_count.emplace_back(v2.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<kalinin_d_vector_dot_product_seq::TestTaskSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
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

  int answer = kalinin_d_vector_dot_product_seq::VectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}

TEST(kalinin_d_vector_dot_product_seq, test_task_run) {
  const int count = 100000000;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = CreateRandomVector(count);
  std::vector<int> v2 = CreateRandomVector(count);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  task_data_seq->inputs_count.emplace_back(v1.size());
  task_data_seq->inputs_count.emplace_back(v2.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<kalinin_d_vector_dot_product_seq::TestTaskSequential>(task_data_seq);
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
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

  int answer = kalinin_d_vector_dot_product_seq::VectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}