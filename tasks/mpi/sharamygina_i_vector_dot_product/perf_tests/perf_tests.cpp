#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/sharamygina_i_vector_dot_product/include/ops_mpi.h"

namespace sharamygina_i_vector_dot_product_mpi {
namespace {
int resulting(const std::vector<int> &v1, const std::vector<int> &v2) {
  int res = 0;
  for (size_t i = 0; i < v1.size(); ++i) {
    res += v1[i] * v2[i];
  }
  return res;
}
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
}  // namespace sharamygina_i_vector_dot_product_mpi

TEST(sharamygina_i_vector_dot_product_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  // Create data
  constexpr int size1 = 12000000;
  constexpr int size2 = 12000000;
  // auto taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);
  std::vector<int> v1(size1);
  std::vector<int> v2(size2);
  int expected_res = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    v1 = sharamygina_i_vector_dot_product_mpi::GetVector(size1);
    v2 = sharamygina_i_vector_dot_product_mpi::GetVector(size2);
    expected_res = sharamygina_i_vector_dot_product_mpi::resulting(v1, v2);

    taskData->inputs_count.emplace_back(size1);
    taskData->inputs_count.emplace_back(size2);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    taskData->outputs_count.emplace_back(received_res.size());
  }

  auto testTask = std::make_shared<sharamygina_i_vector_dot_product_mpi::vector_dot_product_mpi>(taskData);

  ASSERT_EQ(testTask->ValidationImpl(), true);
  testTask->PreProcessingImpl();
  testTask->RunImpl();
  testTask->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(testTask);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(expected_res, received_res[0]);
  }
}

TEST(sharamygina_i_vector_dot_product_mpi, test_task_run) {
  boost::mpi::communicator world;

  // Create data
  constexpr int size1 = 10000000;
  constexpr int size2 = 10000000;
  // auto taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);
  std::vector<int> v1(size1);
  std::vector<int> v2(size2);

  v1 = sharamygina_i_vector_dot_product_mpi::GetVector(size1);
  v2 = sharamygina_i_vector_dot_product_mpi::GetVector(size2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(size1);
    taskData->inputs_count.emplace_back(size2);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    taskData->outputs_count.emplace_back(received_res.size());
  }
  auto testTask = std::make_shared<sharamygina_i_vector_dot_product_mpi::vector_dot_product_mpi>(taskData);

  ASSERT_EQ(testTask->ValidationImpl(), true);
  testTask->PreProcessingImpl();
  testTask->RunImpl();
  testTask->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(testTask);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}
