// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/kalinin_d_vector_dot_product/include/ops_mpi.hpp"

static int offset = 0;
const int count_size_vector = 49000000;

std::vector<int> createRandomVector(int v_size) {
  std::vector<int> vec(v_size);
  std::mt19937 gen;
  gen.seed((unsigned)time(nullptr) + ++offset);
  for (int i = 0; i < v_size; i++) vec[i] = gen() % 100;
  return vec;
}

TEST(kalinin_d_vector_dot_product_mpi, test_pipeline_RunImpl) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;

  std::vector<int> v1 = createRandomVector(count_size_vector);
  std::vector<int> v2 = createRandomVector(count_size_vector);

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

  auto test_task_mpi = std::make_shared<nesterov_a_test_task_mpi::TestTaskMPI>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_RunImplning = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  int answer = kalinin_d_vector_dot_product_mpi::vectorDotProduct(v1, v2);

  //  Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(answer, res[0]);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, test_task_RunImpl) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::vector<int> v1 = createRandomVector(count_size_vector);
  std::vector<int> v2 = createRandomVector(count_size_vector);

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

  auto test_task_mpi = std::make_shared<kalinin_d_vector_dot_product_mpi::TestTaskMPI>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_RunImplning = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // int answer = res[0];
  //   Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->task_RunImpl(perfAttr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(kalinin_d_vector_dot_product_mpi::vectorDotProduct(global_vec[0], global_vec[1]), res[0]);
  }
}