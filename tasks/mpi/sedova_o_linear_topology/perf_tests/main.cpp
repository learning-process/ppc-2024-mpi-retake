#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/sedova_o_linear_topology/include/ops_mpi.hpp"

TEST(sedova_o_linear_topology_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> input;
  bool result = false;
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count = 5000;
    input = sedova_o_linear_topology_mpi::GetRandomVector(count);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data_par->inputs_count.emplace_back(input.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_par->outputs_count.emplace_back(1);
  }

  sedova_o_linear_topology_mpi::TestTaskMPI test_task_parallel(task_data_par);
  ASSERT_EQ(test_task_parallel.ValidationImpl(), true);
  test_task_parallel.PreProcessingImpl();
  test_task_parallel.RunImpl();
  test_task_parallel.PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_data_par);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(result);
  }
}

TEST(sedova_o_linear_topology_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> input;
  bool result = false;
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count = 5000;
    input = sedova_o_linear_topology_mpi::GetRandomVector(count);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data_par->inputs_count.emplace_back(input.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_par->outputs_count.emplace_back(1);
  }

  sedova_o_linear_topology_mpi::TestTaskMPI test_task_parallel(task_data_par);
  ASSERT_EQ(test_task_parallel.ValidationImpl(), true);
  test_task_parallel.PreProcessingImpl();
  test_task_parallel.RunImpl();
  test_task_parallel.PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_data_par);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(result);
  }
}