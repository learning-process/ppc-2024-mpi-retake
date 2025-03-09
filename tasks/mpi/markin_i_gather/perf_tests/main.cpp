#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/markin_i_gather/include/ops_mpi.hpp"

namespace markin_i_gather {

template <typename T>
void GenerateTestData(int size, std::vector<T>& data) {
  data.resize(size);
  for (int i = 0; i < size; ++i) {
    data[i] = static_cast<T>(i);
  }
}

}  // namespace markin_i_gather

TEST(markin_i_gather, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int rank = world.rank();
  const int root = 0;
  const int data_size = 1000;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  std::vector<int> send_data;
  markin_i_gather::GenerateTestData(data_size, send_data);
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(send_data.data()));
  task_data_mpi->inputs_count.emplace_back(send_data.size());

  auto test_task_mpi = std::make_shared<markin_i_gather::MyGatherMpiTask>(task_data_mpi, world);
  test_task_mpi->SetRoot(root, world);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}

TEST(markin_i_gather, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int rank = world.rank();
  const int root = 0;
  const int data_size = 1000;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  std::vector<int> send_data;
  markin_i_gather::GenerateTestData(data_size, send_data);
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(send_data.data()));
  task_data_mpi->inputs_count.emplace_back(send_data.size());

  auto test_task_mpi = std::make_shared<markin_i_gather::MyGatherMpiTask>(task_data_mpi, world);
  test_task_mpi->SetRoot(root, world);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }
}