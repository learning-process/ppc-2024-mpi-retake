#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/chernova_n_topology_ring/include/ops_mpi.hpp"

namespace chernova_n_topology_ring_mpi {

std::vector<char> GenerateDataPerf(int k) {
  const std::string words[] = {"one", "two", "three"};
  const int word_array_size = sizeof(words) / sizeof(words[0]);

  std::string result;

  for (int i = 0; i < k; ++i) {
    result += words[i % word_array_size];
    if (i < k - 1) {
      result += ' ';
    }
  }

  return std::vector<char>(result.begin(), result.end());
}

}  // namespace chernova_n_topology_ring_mpi

TEST(chernova_n_topology_ring_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int k = 100000;
  std::vector<char> test_data_parallel = chernova_n_topology_ring_mpi::GenerateDataPerf(k);
  std::vector<char> in = test_data_parallel;
  const int n = in.size();
  std::vector<char> out_vec(n);
  std::vector<int> out_process;

  auto task_data_parallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size() + 1);
    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(in.data())));
    task_data_parallel->inputs_count.emplace_back(in.size());
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_vec.data()));
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_process.data()));
    task_data_parallel->outputs_count.reserve(2);
  }

  auto test_mpi_task_parallel = std::make_shared<chernova_n_topology_ring_mpi::TestMPITaskParallel>(task_data_parallel);
  ASSERT_EQ(test_mpi_task_parallel->Validation(), true);
  test_mpi_task_parallel->PreProcessing();
  test_mpi_task_parallel->Run();
  test_mpi_task_parallel->PostProcessing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(true, std::equal(in.begin(), in.end(), out_vec.begin(), out_vec.end()));
  }
}

TEST(chernova_n_topology_ring_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int k = 100000;
  std::vector<char> test_data_parallel = chernova_n_topology_ring_mpi::GenerateDataPerf(k);
  std::vector<char> in = test_data_parallel;
  const int n = in.size();
  std::vector<char> out_vec(n);
  std::vector<int> out_process;

  auto task_data_parallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size() + 1);
    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(in.data())));
    task_data_parallel->inputs_count.emplace_back(in.size());
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_vec.data()));
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_process.data()));
    task_data_parallel->outputs_count.reserve(2);
  }

  auto test_mpi_task_parallel = std::make_shared<chernova_n_topology_ring_mpi::TestMPITaskParallel>(task_data_parallel);
  ASSERT_EQ(test_mpi_task_parallel->Validation(), true);
  test_mpi_task_parallel->PreProcessing();
  test_mpi_task_parallel->Run();
  test_mpi_task_parallel->PostProcessing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(true, std::equal(in.begin(), in.end(), out_vec.begin(), out_vec.end()));
  }
}