#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <climits>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/sedova_o_linear_topology/include/ops_mpi.hpp"

namespace {
std::vector<int> sedova_o_linear_topology_mpi::GetRandomVector(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(1, 500);
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = distrib(gen);
  }
  return vec;
}
}  // namespace

TEST(sedova_o_linear_topology_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> input;
  bool result = false;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 50000000;
    input = sedova_o_linear_topology_mpi::GetRandomVector(count);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data_par->inputs_count.emplace_back(input.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_par->outputs_count.emplace_back(1);
  }

  auto task_data_parallel = std::make_shared<sedova_o_linear_topology_mpi::TestTaskMPI>(task_data_par);
  ASSERT_EQ(task_data_parallel->ValidationImpl(), true);
  task_data_parallel->PreProcessingImpl();
  task_data_parallel->RunImpl();
  task_data_parallel->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10000;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_data_parallel);
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

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 50000000;
    input = sedova_o_linear_topology_mpi::GetRandomVector(count);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data_par->inputs_count.emplace_back(input.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
    task_data_par->outputs_count.emplace_back(1);
  }

  auto task_data_parallel = std::make_shared<sedova_o_linear_topology_mpi::TestTaskMPI>(task_data_par);
  ASSERT_EQ(task_data_parallel->ValidationImpl(), true);
  task_data_parallel->PreProcessingImpl();
  task_data_parallel->RunImpl();
  task_data_parallel->PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10000;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_data_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(result);
  }
}