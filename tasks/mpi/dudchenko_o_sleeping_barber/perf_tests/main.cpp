#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/dudchenko_o_sleeping_barber/include/ops_mpi.hpp"

TEST(dudchenko_o_sleeping_barber_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  if (world.size() < 3) {
    return;
  }

  const int max_waiting_chairs = 3;
  bool barber_busy = false;
  std::vector<int> global_res(1, 0);
  int num_clients = std::max(1, world.size() - 2);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs_count = {max_waiting_chairs, static_cast<unsigned int>(barber_busy)};
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
  }

  auto test_mpi_task_parallel = std::make_shared<dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber>(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel->ValidationImpl());
  test_mpi_task_parallel->PreProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = num_clients;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  test_mpi_task_parallel->RunImpl();
  test_mpi_task_parallel->PostProcessingImpl();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(1, global_res[0]);
  }
}

TEST(dudchenko_o_sleeping_barber_mpi, test_task_run) {
  boost::mpi::communicator world;
  if (world.size() < 3) {
    return;
  }

  const int max_waiting_chairs = 3;
  std::vector<int> global_res(1, 0);
  bool barber_busy = false;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  int num_clients = std::max(1, world.size() - 2);

  if (world.rank() == 0) {
    task_data_par->inputs_count = {max_waiting_chairs, static_cast<unsigned int>(barber_busy)};
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    task_data_par->outputs_count.emplace_back(global_res.size());
  }

  auto test_mpi_task_parallel = std::make_shared<dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber>(task_data_par);
  ASSERT_TRUE(test_mpi_task_parallel->ValidationImpl());
  test_mpi_task_parallel->PreProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = num_clients;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  test_mpi_task_parallel->RunImpl();
  test_mpi_task_parallel->PostProcessingImpl();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(1, global_res[0]);
  }
}