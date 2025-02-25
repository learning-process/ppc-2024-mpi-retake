// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/kalinin_d_vector_dot_product/include/ops_mpi.hpp"

namespace {
int offset = 0;
}  // namespace
const int kCountSizeVector = 42000000;

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

TEST(kalinin_d_vector_dot_product_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;

  std::vector<int> v1 = CreateRandomVector(kCountSizeVector);
  std::vector<int> v2 = CreateRandomVector(kCountSizeVector);

  std::vector<int32_t> res(1, 0);
  global_vec = {v1, v2};

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }

  auto test_task_mpi = std::make_shared<kalinin_d_vector_dot_product_mpi::TestMPITaskParallel>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  int answer = kalinin_d_vector_dot_product_mpi::VectorDotProduct(v1, v2);

  //  Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(answer, res[0]);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::vector<int> v1 = CreateRandomVector(kCountSizeVector);
  std::vector<int> v2 = CreateRandomVector(kCountSizeVector);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  global_vec = {v1, v2};

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }

  auto test_task_mpi = std::make_shared<kalinin_d_vector_dot_product_mpi::TestMPITaskParallel>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  //   Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(kalinin_d_vector_dot_product_mpi::VectorDotProduct(global_vec[0], global_vec[1]), res[0]);
  }
}