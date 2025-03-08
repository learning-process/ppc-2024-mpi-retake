// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/makhov_m_ring_topology/include/ops_mpi.hpp"

TEST(makhov_m_ring_topology_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  size_t size = 10000000;
  std::vector<int32_t> input_vector(size, 1);
  std::vector<int32_t> output_vector(size);
  std::vector<int32_t> sequence(world.size() + 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequence.data()));
    task_data_par->outputs_count.emplace_back(size);
    task_data_par->outputs_count.emplace_back(world.size() + 1);
  }

  auto test_mpi_task_parallel = std::make_shared<makhov_m_ring_topology::TestMPITaskParallel>(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel->ValidationImpl());
  test_mpi_task_parallel->PreProcessingImpl();
  test_mpi_task_parallel->RunImpl();
  test_mpi_task_parallel->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(makhov_m_ring_topology_mpi, test_task_run) {
  boost::mpi::communicator world;
  size_t size = 10000000;
  std::vector<int32_t> input_vector(size, 1);
  std::vector<int32_t> output_vector(size);
  std::vector<int32_t> sequence(world.size() + 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequence.data()));
    task_data_par->outputs_count.emplace_back(size);
    task_data_par->outputs_count.emplace_back(world.size() + 1);
  }

  auto test_mpi_task_parallel = std::make_shared<makhov_m_ring_topology::TestMPITaskParallel>(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel->ValidationImpl());
  test_mpi_task_parallel->PreProcessingImpl();
  test_mpi_task_parallel->RunImpl();
  test_mpi_task_parallel->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}
