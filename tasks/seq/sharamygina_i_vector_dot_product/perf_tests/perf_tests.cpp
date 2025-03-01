#include <gtest/gtest.h>

#include <chrono>
#include <limits>
#include <random>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sharamygina_i_vector_dot_product/include/ops_seq.h"

namespace sharamygina_i_vector_dot_product_seq {
namespace {
std::vector<int> GetVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> v(size);
  for (int i = 0; i < size; i++) {
    v[i] = gen() % 320 + gen() % 11;
  }
  return v;
}
}  // namespace
}  // namespace sharamygina_i_vector_dot_product_seq

TEST(sharamygina_i_vector_dot_product_seq, LargeImage) {
  constexpr int size1 = 10000000;
  constexpr int size2 = 10000000;
  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs_count.emplace_back(size1);
  taskData->inputs_count.emplace_back(size2);

  std::vector<int> received_res(1);
  std::vector<int> v1(size1);
  std::vector<int> v2(size2);
  v1 = sharamygina_i_vector_dot_product_seq::GetVector(size1);
  v2 = sharamygina_i_vector_dot_product_seq::GetVector(size2);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  taskData->outputs_count.emplace_back(received_res.size());

  auto task = std::make_shared<sharamygina_i_vector_dot_product_seq::vector_dot_product_seq>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;

  const auto start = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&]() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->PipelineRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
}

TEST(sharamygina_i_vector_dot_product_seq, LargeImageRun) {
  constexpr int size1 = 10000000;
  constexpr int size2 = 10000000;
  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs_count.emplace_back(size1);
  taskData->inputs_count.emplace_back(size2);

  std::vector<int> received_res(1);
  std::vector<int> v1(size1);
  std::vector<int> v2(size2);
  v1 = sharamygina_i_vector_dot_product_seq::GetVector(size1);
  v2 = sharamygina_i_vector_dot_product_seq::GetVector(size2);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  taskData->outputs_count.emplace_back(received_res.size());

  auto task = std::make_shared<sharamygina_i_vector_dot_product_seq::vector_dot_product_seq>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;

  const auto start = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&]() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->TaskRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
}
