#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/fomin_v_generalized_scatter/include/ops_mpi.hpp"

TEST(fomin_v_generalized_scatter, test_task_run) {
  boost::mpi::communicator world;
  int local_size = 10;
  std::vector<int> local_output(local_size, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_input.data()));
    task_data->inputs_count.emplace_back(global_input.size());
  }
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_output.data()));
  task_data->outputs_count.emplace_back(local_output.size());

  auto test_task = std::make_shared<fomin_v_generalized_scatter::GeneralizedScatterTestParallel>(task_data);
  ASSERT_TRUE(test_task->validation());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  // Verify results
  std::vector<int> received_data;
  if (world.rank() == 0) {
    received_data.resize(world.size() * local_size);
  }

  MPI_Gather(local_output.data(), local_size, MPI_INT, received_data.data(), local_size, MPI_INT, 0, world);

  if (world.rank() == 0) {
    for (int i = 0; i < world.size(); ++i) {
      for (int j = 0; j < local_size; ++j) {
        ASSERT_EQ(received_data[i * local_size + j], global_input[i * local_size + j]);
      }
    }
  }
}

TEST(fomin_v_generalized_scatter, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> local_output(local_size, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_input.data()));
    task_data->inputs_count.emplace_back(global_input.size());
  }
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_output.data()));
  task_data->outputs_count.emplace_back(local_output.size());

  auto test_task = std::make_shared<fomin_v_generalized_scatter::GeneralizedScatterTestParallel>(task_data);
  ASSERT_TRUE(test_task->validation());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  // Verify results
  std::vector<int> received_data;
  if (world.rank() == 0) {
    received_data.resize(world.size() * local_size);
  }

  MPI_Gather(local_output.data(), local_size, MPI_INT, received_data.data(), local_size, MPI_INT, 0, world);

  if (world.rank() == 0) {
    for (int i = 0; i < world.size(); ++i) {
      for (int j = 0; j < local_size; ++j) {
        ASSERT_EQ(received_data[i * local_size + j], global_input[i * local_size + j]);
      }
    }
  }
}

// namespace fomin_v_generalized_scatter
