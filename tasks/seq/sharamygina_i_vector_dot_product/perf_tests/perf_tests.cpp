#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sharamygina_i_vector_dot_product/include/ops_seq.h"

namespace sharamygina_i_vector_dot_product_seq {
namespace {
std::vector<int> GetVector(unsigned int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> v(size);
  for (unsigned int i = 0; i < size; i++) {
    v[i] = static_cast<int>((gen() % 320) - (gen() % 97));
  }
  return v;
}
}  // namespace
}  // namespace sharamygina_i_vector_dot_product_seq

TEST(sharamygina_i_vector_dot_product_seq, LargeImage) {
  constexpr unsigned int kLenght = 12000000;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs_count.emplace_back(kLenght);
  task_data->inputs_count.emplace_back(kLenght);

  std::vector<int> received_res(1);
  std::vector<int> v1(kLenght);
  std::vector<int> v2(kLenght);
  v1 = sharamygina_i_vector_dot_product_seq::GetVector(kLenght);
  v2 = sharamygina_i_vector_dot_product_seq::GetVector(kLenght);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  task_data->outputs_count.emplace_back(received_res.size());

  auto task = std::make_shared<sharamygina_i_vector_dot_product_seq::VectorDotProductSeq>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1;

  const auto start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(sharamygina_i_vector_dot_product_seq, LargeImageRun) {
  constexpr unsigned int kLenght = 12000000;
  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs_count.emplace_back(kLenght);
  task_data->inputs_count.emplace_back(kLenght);

  std::vector<int> received_res(1);
  std::vector<int> v1(kLenght);
  std::vector<int> v2(kLenght);
  v1 = sharamygina_i_vector_dot_product_seq::GetVector(kLenght);
  v2 = sharamygina_i_vector_dot_product_seq::GetVector(kLenght);

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(received_res.data()));
  task_data->outputs_count.emplace_back(received_res.size());

  auto task = std::make_shared<sharamygina_i_vector_dot_product_seq::VectorDotProductSeq>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1;

  const auto start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
