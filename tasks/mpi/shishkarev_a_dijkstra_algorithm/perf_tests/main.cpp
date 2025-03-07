// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/shishkarev_a_dijkstra_algorithm/include/ops_mpi.hpp"

TEST(shishkarev_a_dijkstra_algorithm_mpi, test_PipelineRun) {
  boost::mpi::communicator world;
  int count_size_vector = 5000;
  int st = 0;
  std::vector<int> global_matrix(count_size_vector * count_size_vector, 3);
  std::vector<int32_t> global_path(count_size_vector, 3);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      global_matrix[(i * count_size_vector) + i] = 0;
    }
    global_path[0] = 0;
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(count_size_vector);
    task_data_par->inputs_count.emplace_back(st);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_path.data()));
    task_data_par->outputs_count.emplace_back(global_path.size());
  }

  auto test_mpi_task_parallel =
      std::make_shared<shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel>(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel->ValidationImpl(), true);
  test_mpi_task_parallel->PreProcessingImpl();
  test_mpi_task_parallel->RunImpl();
  test_mpi_task_parallel->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(3, global_path[3]);
  }
}

TEST(shishkarev_a_dijkstra_algorithm_mpi, test_task_run) {
  boost::mpi::communicator world;
  int count_size_vector = 5000;
  int st = 5;
  std::vector<int> global_matrix(count_size_vector * count_size_vector, 3);
  std::vector<int32_t> global_path(count_size_vector, 3);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      global_matrix[(i * count_size_vector) + i] = 0;
    }
    global_path[0] = 0;
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    task_data_par->inputs_count.emplace_back(global_matrix.size());
    task_data_par->inputs_count.emplace_back(count_size_vector);
    task_data_par->inputs_count.emplace_back(st);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_path.data()));
    task_data_par->outputs_count.emplace_back(global_path.size());
  }

  auto test_mpi_task_parallel =
      std::make_shared<shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel>(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel->ValidationImpl(), true);
  test_mpi_task_parallel->PreProcessingImpl();
  test_mpi_task_parallel->RunImpl();
  test_mpi_task_parallel->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(3, global_path[3]);
  }
}