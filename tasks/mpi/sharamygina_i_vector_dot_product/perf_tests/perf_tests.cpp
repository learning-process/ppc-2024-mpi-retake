#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/sharamygina_i_vector_dot_product/include/ops_mpi.h"

namespace sharamygina_i_vector_dot_product_mpi {
namespace {
int Resulting(const std::vector<int> &v1, const std::vector<int> &v2) {
  int res = 0;
  for (unsigned int i = 0; i < v1.size(); ++i) {
    res += v1[i] * v2[i];
  }
  return res;
}
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
}  // namespace sharamygina_i_vector_dot_product_mpi

TEST(sharamygina_i_vector_dot_product_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  // Create data
  constexpr unsigned int kLenght = 12000000;

  std::vector<int> received_res(1);
  std::vector<int> v1(kLenght);
  std::vector<int> v2(kLenght);
  int expected_res = 0;

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    v1 = sharamygina_i_vector_dot_product_mpi::GetVector(kLenght);
    v2 = sharamygina_i_vector_dot_product_mpi::GetVector(kLenght);
    expected_res = sharamygina_i_vector_dot_product_mpi::Resulting(v1, v2);

    task_data->inputs_count.emplace_back(kLenght);
    task_data->inputs_count.emplace_back(kLenght);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    task_data->outputs_count.emplace_back(received_res.size());
  }

  auto test_task = std::make_shared<sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi>(task_data);

  ASSERT_EQ(test_task->ValidationImpl(), true);
  test_task->PreProcessingImpl();
  test_task->RunImpl();
  test_task->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(expected_res, received_res[0]);
  }
}

TEST(sharamygina_i_vector_dot_product_mpi, test_task_run) {
  boost::mpi::communicator world;

  // Create data
  constexpr unsigned int kLenght = 12000000;

  std::vector<int> received_res(1);
  std::vector<int> v1(kLenght);
  std::vector<int> v2(kLenght);

  v1 = sharamygina_i_vector_dot_product_mpi::GetVector(kLenght);
  v2 = sharamygina_i_vector_dot_product_mpi::GetVector(kLenght);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(kLenght);
    task_data->inputs_count.emplace_back(kLenght);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    task_data->outputs_count.emplace_back(received_res.size());
  }
  auto test_task = std::make_shared<sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi>(task_data);

  ASSERT_EQ(test_task->ValidationImpl(), true);
  test_task->PreProcessingImpl();
  test_task->RunImpl();
  test_task->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}
